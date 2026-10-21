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

class ImageDataset(Dataset):
    def __init__(self, image_dir: Path, labeled_image_paths: List[Tuple[Path, Any]], label_encodings, transforms=None):
        super().__init__()
        self.image_dir = image_dir
        self.labeled_image_paths = labeled_image_paths
        self.label_encodings = label_encodings
        self.transforms = transforms
    
    def set_transforms(self, transforms):
        self.transforms = transforms
    
    def __len__(self):
        return len(self.labeled_image_paths)
    
    def __getitem__(self, index):
        image_name, label = self.labeled_image_paths[index]
        image_path = self.image_dir / image_name
        image = decode_image(image_path)
        if self.transforms is not None:
            image = self.transforms(image)
        if not isinstance(label, int):
            label = self.label_encodings[label]
        return image, label

    def get_n_classes(self):
        # Different labels can have the same encoding, this
        # makes sure we count the distinct number of encoded labels
        return len(set(self.label_encodings.values()))


class MultiTaskDataset(Dataset):
    def __init__(self, image_dir, image_names, metadata, tasks, task_encodings, task_weights, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.tasks = tasks
        self.task_encodings = task_encodings
        self.task_weights = task_weights
        self.image_names = image_names
        self.image_labels = {}
        for record in metadata.to_dict('records'):
            image = record['image']
            image_labels = {}
            for task, encodings in sorted(task_encodings.items()):
                image_task_value = record[task]
                try:
                    encoded_label = encodings[image_task_value]
                except KeyError:
                     continue
                image_labels[task] = encoded_label
            self.image_labels[image] = image_labels
            
        super().__init__()

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = self.image_dir / image_name
        image = decode_image(image_path)
        if self.transforms is not None:
            image = self.transforms(image)
        labels = self.image_labels[image_name]    
        return image, labels
        
    def set_transforms(self, transforms):
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_names)



def multiclass_collator(batch, *, tasks, task_encodings):
    label_arrays = [[] for t in tasks]
    label_masks = [[] for t in tasks]
    images = []
    for image, encoded_labels in batch:
        images.append(image)
        for i, task in enumerate(tasks):
            if task in encoded_labels:
                label_arrays[i].append(encoded_labels[task])
                label_masks[i].append(1)
            else:
                label_arrays[i].append(0)
                label_masks[i].append(0)
    images = torch.stack(images, dim=0)
    labels = [torch.tensor(l) for l in label_arrays]
    masks = [torch.tensor(m) for m in label_masks]
    return images, labels, masks
    
