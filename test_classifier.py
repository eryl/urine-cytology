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

from models import Classifier, TimmCNNEncoder, PredictionHead, get_transforms


class ImageDataset(Dataset):
    def __init__(self, images, transforms=None):
        super().__init__()
        self.images = images
        self.transforms = transforms
    
    def set_transforms(self, transforms):
        self.transforms = transforms
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = self.images[index]
        image = decode_image(image_path)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, image_path.name
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_image_dir', type=Path)
    parser.add_argument('experiment_dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--frozen-epochs', type=int, default=20)
    parser.add_argument('--unfrozen-epochs', type=int, default=1000)
    parser.add_argument('--random-seed', type=int, default=1729)
    parser.add_argument('--model-name', choices=('uni_v1', 'resnet50_trunc'), default='resnet50_trunc')
    
    args = parser.parse_args()
    
    test_images = sorted(args.test_image_dir.glob('*.png'))
    
    eval_transforms, train_transforms = get_transforms(args.model_name)
    
    test_dataset = ImageDataset(test_images, transforms=eval_transforms)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 shuffle=False, 
                                 drop_last=False, 
                                 persistent_workers=args.num_workers > 0)
    
    device = torch.device(args.device)
    
    rng = random.Random(args.random_seed)
    checkpoints = sorted(args.experiment_dir.glob('**/*.ckpt'))
    # Shuffle just so that the order we get the predictions in will 
    # be less biased. Good if we don't have time to do all the predictions 
    # but still want a reasonably random predictions from models which have 
    # been trained on more varied data. Taking them in sorted order takes them 
    # according to fold, which will have a training sample bias (e.g the first 10 
    # models share most of their training data)
    rng.shuffle(checkpoints)  
    
    predictions_dir = args.experiment_dir / f'test_predictions_{args.test_image_dir.name}'
    predictions_dir.mkdir(exist_ok=True, parents=True)
    
    for i, checkpoint in enumerate(tqdm(checkpoints, desc="Checkpoints")):
        model = Classifier.load_from_checkpoint(checkpoint)
        model.to(device)
        model.eval()
        with torch.inference_mode():
            image_names = []
            logits = []
            probabilities = []
            for test_batch in tqdm(test_dataloader, desc="Test batches"):
                images, batch_names = test_batch
                batch_logits = model(images.to(device))
                if len(batch_logits.shape) > 1:
                    batch_probabilities = torch.softmax(batch_logits, dim=-1)
                else:
                    batch_probabilities = torch.sigmoid(batch_logits)
                image_names.extend(batch_names)
                logits.append(batch_logits.cpu().numpy())
                probabilities.append(batch_probabilities.cpu().numpy())

            logits_arr = np.concatenate(logits)
            probabilities_arr = np.concatenate(probabilities)
            
            data = dict(image_names=image_names)
            if len(logits_arr.shape) > 1:
                for j in range(logits_arr.shape[-1]):
                    data[f"logits_{j}"] = logits_arr[:, j]
                    data[f"probabilities_{j}"] = probabilities_arr[:, j]
            else:
                data["logits"] = logits_arr.squeeze()
                data["probabilities"] = probabilities_arr.squeeze()
            
            df = pd.DataFrame(data)
            df['model_path'] = str(checkpoint)
            output_path = predictions_dir / f'predictions_{i:02}.csv'
            df.to_csv(output_path, index=False)
                
    

if __name__ == '__main__':
    main()
    