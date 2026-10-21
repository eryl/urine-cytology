import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Set, Tuple
import datetime
import math
import json

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import lightning
from lightning.pytorch.callbacks import ModelCheckpoint, BaseFinetuning
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

from torchvision.io import decode_image
import torch
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F

import os
from functools import partial
import timm
import torch
from tqdm import trange, tqdm

from datasets import MultiTaskDataset, multiclass_collator
from models import get_encoder, Classifier, MultiTaskClassifier, FeatureExtractorFreezeUnfreeze
    
@dataclass
class TrainingArgs:
    optimization_batch_size: int
    device_batch_size: int
    random_seed: int
    num_workers: int
    frozen_epochs: int
    unfrozen_epochs: int
    lr: float
    device: str
    hidden_dim: int
    model_name: Literal['uni_v1', 'resnet50_trunc', 'efficientnet_s', 'efficientnet_m']
    metadata: Optional[dict]

ENCODING_OVERRIDES = {'nc_ratio': {'NC ratio < 0.5': 0, 'NC ratio 0.5 < 0.7': 1, 'NC ratio > 0.7': 2},
                      'quality': {'Useless': 0, 'Poor quality': 1, 'Useful': 2}}

def collate_task_mapping(task_metadata: pd.DataFrame):
    task_classes = {task: set() for task in task_metadata.columns}
    for record in task_metadata.to_dict('records'):
        for k, v in record.items():
            try:
                if math.isnan(v):
                    continue
            except TypeError:
                pass
            task_classes[k].add(v)

    task_label_encodings = {}
    for task, task_values in task_classes.items():
        if task in ENCODING_OVERRIDES:
            task_label_encodings[task] = ENCODING_OVERRIDES[task]
        else: 
            task_label_encodings[task] = {label: i for i, label in enumerate(sorted(task_values))}
    
    return task_label_encodings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=Path)
    parser.add_argument('split_file', type=Path)
    parser.add_argument('metadata_file', type=Path)
    parser.add_argument('experiment_name', help="What to name the experiment")
    parser.add_argument('--output-dir', type=Path, default=Path('experiments'))
    parser.add_argument('--random-seed', type=int, default=1729)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--num-workers', type=int, default=14)
    parser.add_argument('--optimization-batch-size', type=int, default=128)
    parser.add_argument('--device-batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--frozen-epochs', type=int, default=20)
    parser.add_argument('--unfrozen-epochs', type=int, default=1000)
    parser.add_argument('--model-name', choices=('uni_v1', 'resnet50_trunc', 'efficientnet_s', 'efficientnet_m'), default='efficientnet_m')
    parser.add_argument('--tasks', help="What columns from the metadata CSV to use as targets", nargs='+', default=["quality", 
                                                                                                                    "nc_ratio", 
                                                                                                                    "hyperchromasia", 
                                                                                                                    "coarse_chromatin", 
                                                                                                                    "irregular_chromatin", 
                                                                                                                    "reactive_atypia",
                                                                                                                    "is_group",
                                                                                                                    "inflammatory",
                                                                                                                    "not_a_cell",
                                                                                                                    "no_cytoplasma"])
    parser.add_argument('--task-weights', help="How much weight each task should be assigned in the loss function, should be one floating point value per task in `--tasks`", nargs='+', type=float)
    
    args = parser.parse_args()
    
    #rng = random.Random(args.random_seed)    
    rng = np.random.default_rng(args.random_seed)
    
    with open(args.split_file) as fp:
        split_definitions = json.load(fp)
    
    metadata = pd.read_csv(args.metadata_file)
    tasks = args.tasks
    
    for task in tasks:
        if task not in metadata.columns:
            raise ValueError(f"Task {task} is not in metadata columns {metadata.columns}")
    
    n_tasks = len(tasks)
    task_weights = args.task_weights
    if task_weights is None:
        task_weights = [1.0/n_tasks]*n_tasks
    elif len(task_weights) != len(tasks):
        raise ValueError(f"Task weights {task_weights} differes in length from tasks: {tasks}")
    
    task_weights = dict(zip(tasks, task_weights))
    tasks = sorted(tasks) # Now that we've established the weights, we are free to reorder the tasks. We'll make a canonical order which will be used everywhere, which is the sorted order

    task_metadata = metadata[tasks]
    task_class_mappings = collate_task_mapping(task_metadata)
    #device = torch.device(args.device)
    experiment_class_dir: Path = args.output_dir / args.experiment_name / "multitask"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    experiment_root =  experiment_class_dir / f"{timestamp}"
    experiment_root.mkdir(exist_ok=False, parents=True)

    latest_experiment = experiment_class_dir / 'latest_experiment'
    if latest_experiment.exists():
        latest_experiment.unlink()
    latest_experiment.symlink_to(experiment_root.relative_to(experiment_class_dir))
    
    # Since going through the folds in order introduces a sampling bias (e.g. they're using very 
    # similar training data). By shuffling, the order in which we train the models is unbiased, 
    # which is good if we don't want to train all of them
    folds = [(outer_fold_i, inner_fold_i) for outer_fold_i, inner_folds in split_definitions['splits'].items() for inner_fold_i in inner_folds.keys()]
    rng.shuffle(folds)
    
    for (outer_fold_i, inner_fold_i) in folds:
        splits = split_definitions['splits'][outer_fold_i][inner_fold_i]
        train_images, train_split_label = zip(*splits['train'])
        train_dataset = MultiTaskDataset(args.image_dir, train_images, metadata, tasks, task_class_mappings, task_weights)
        dev_images, dev_split_labels = zip(*splits['dev'])
        dev_dataset = MultiTaskDataset(args.image_dir, dev_images, metadata, tasks, task_class_mappings, task_weights)
        test_images, test_split_labels = zip(*splits['test'])
        test_dataset = MultiTaskDataset(args.image_dir, test_images, metadata, tasks, task_class_mappings, task_weights)
        
        output_dir = experiment_root / f"{outer_fold_i}" / f"{inner_fold_i}"
        output_dir.mkdir(parents=True)
        
        training_args = TrainingArgs(optimization_batch_size=args.optimization_batch_size, 
                                    device_batch_size=args.device_batch_size,
                                    num_workers=args.num_workers, 
                                    random_seed=rng, 
                                    frozen_epochs=args.frozen_epochs, 
                                    unfrozen_epochs=args.unfrozen_epochs, 
                                    lr=args.learning_rate, 
                                    device=args.device,
                                    model_name=args.model_name,
                                    hidden_dim=args.hidden_dim,
                                    metadata={'train_images': train_images, 'dev_images': dev_images, 'test_images': test_images, 'class_mappings': task_class_mappings})
        training_worker(train_dataset, dev_dataset, test_dataset, training_args, output_dir=output_dir)

    

def training_worker(train_dataset: MultiTaskDataset, dev_dataset: MultiTaskDataset, test_dataset: MultiTaskDataset, args: TrainingArgs, output_dir):
    model, eval_transforms, train_transforms, output_dim = get_encoder(model_name=args.model_name)
    task_class_mappings = train_dataset.task_encodings
    tasks = train_dataset.tasks
    task_weights = train_dataset.task_weights
    classifier = MultiTaskClassifier(model, output_dim, args.hidden_dim, tasks, task_class_mappings, task_weights)
    train_dataset.set_transforms(train_transforms)
    dev_dataset.set_transforms(eval_transforms)
    test_dataset.set_transforms(eval_transforms)
    
    #classifier.to(device)
    collator = partial(multiclass_collator, tasks=tasks, task_encodings=train_dataset.task_encodings)

    dataloader_train = DataLoader(
        dataset=train_dataset,
        batch_size=args.device_batch_size,
        num_workers=args.num_workers,
        #sampler=train_sampler,   
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator,
        persistent_workers=args.num_workers > 0,
    )
    
    dataloader_dev = DataLoader(
        dataset=dev_dataset,
        batch_size=args.device_batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        shuffle=False,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    
    accumulate_gradients = args.optimization_batch_size // args.device_batch_size
    
    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    logger = TensorBoardLogger(save_dir=output_dir, version=1, name="lightning_logs")
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    unfreeze_callback = FeatureExtractorFreezeUnfreeze(args.frozen_epochs)
    trainer = lightning.Trainer(accelerator=args.device, 
                                callbacks=[unfreeze_callback, checkpoint_callback, early_stopping], 
                                logger=logger, 
                                max_epochs=args.frozen_epochs+args.unfrozen_epochs, 
                                precision="bf16-mixed",
                                log_every_n_steps=4,
                                accumulate_grad_batches=accumulate_gradients)
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as fp:
        json.dump(args.metadata, fp, indent=2)
    
    #tuner = Tuner(trainer)
    #lr_finder = tuner.lr_find(classifier, dataloader_train, dataloader_dev)
    #new_lr = lr_finder.suggestion()
    #classifier.learning_rate = new_lr
    trainer.fit(classifier, dataloader_train, dataloader_dev)
    
    test_results = trainer.test(ckpt_path='best', 
                                #classifier,  # If you don't pass a model, it'll use the best model checkpoint from previous trainer.fit 
                                dataloaders=DataLoader(test_dataset, 
                                                       collate_fn=collator, 
                                                       batch_size=args.device_batch_size, 
                                                       num_workers=args.num_workers, 
                                                       drop_last=False)
                                )

    
    
if __name__ == '__main__':
    main()
    