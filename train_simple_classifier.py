import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Set, Tuple
import datetime

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

from datasets import ImageDataset
from models import get_encoder, Classifier, FeatureExtractorFreezeUnfreeze
    
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=Path)
    parser.add_argument('split_file', type=Path)
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
    
    args = parser.parse_args()
    
    #rng = random.Random(args.random_seed)    
    rng = np.random.default_rng(args.random_seed)
    
    with open(args.split_file) as fp:
        split_definitions = json.load(fp)
    
    label_encodings = split_definitions['label_encodings']
    label_column = split_definitions['label_column']
    #device = torch.device(args.device)
    experiment_class_dir: Path = args.output_dir / label_column
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
        train_splits = splits['train']
        train_dataset = ImageDataset(args.image_dir, train_splits, label_encodings)
        dev_splits = splits['dev']
        dev_dataset = ImageDataset(args.image_dir, dev_splits, label_encodings)
        test_splits = splits['test']
        test_dataset = ImageDataset(args.image_dir, test_splits, label_encodings)
        
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
                                    hidden_dim=args.hidden_dim)
        training_worker(train_dataset, dev_dataset, test_dataset, training_args, output_dir=output_dir)

    

def training_worker(train_dataset: ImageDataset, dev_dataset: ImageDataset, test_dataset: ImageDataset, args: TrainingArgs, output_dir):
    model, eval_transforms, train_transforms, output_dim = get_encoder(model_name=args.model_name)
    n_classes = train_dataset.get_n_classes()
    classifier = Classifier(model, output_dim, args.hidden_dim, n_classes)
    train_dataset.set_transforms(train_transforms)
    dev_dataset.set_transforms(eval_transforms)
    test_dataset.set_transforms(eval_transforms)
    
    #classifier.to(device)
    
    dataloader_train = DataLoader(
        dataset=train_dataset,
        batch_size=args.device_batch_size,
        num_workers=args.num_workers,
        #sampler=train_sampler,   
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    
    dataloader_dev = DataLoader(
        dataset=dev_dataset,
        batch_size=args.device_batch_size,
        num_workers=args.num_workers,
        #collate_fn=collate_hdf5_graphs,
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
    
    #tuner = Tuner(trainer)
    #lr_finder = tuner.lr_find(classifier, dataloader_train, dataloader_dev)
    #new_lr = lr_finder.suggestion()
    #classifier.learning_rate = new_lr
    trainer.fit(classifier, dataloader_train, dataloader_dev)
    
    test_results = trainer.test(ckpt_path='best', 
                                       #classifier,  # If you don't pass a model, it'll use the best model checkpoint from previous trainer.fit 
                                       dataloaders=DataLoader(test_dataset))

    
    
if __name__ == '__main__':
    main()
    