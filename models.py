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




IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]

MODEL2CONSTANTS = {
	"resnet50_trunc": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD,
        "output_dim": 1024,
	},
	"uni_v1":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD,
        "output_dim": 1024,
	},
	"conch_v1":
	{
		"mean": OPENAI_MEAN,
		"std": OPENAI_STD,
        "output_dim": 1024,
	},
    "conch_v1_5":
    {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD,
        "output_dim": 512,
	},
    "dino_v2":
    {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD,
        "output_dim": 512,
	}
}

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH


class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k', cache_dir='timm_cache',
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0}, 
                 pool: bool = True):
        super().__init__()
        assert kwargs.get('pretrained', False), 'only pretrained models are supported'
        self.model = timm.create_model(model_name, cache_dir=cache_dir, **kwargs)
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out


class TorchVisionEfficientNet(torch.nn.Module):
    def __init__(self, model_name, *args, **kwargs):
        from torchvision.models.efficientnet import efficientnet_v2_s, efficientnet_v2_m, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights
        super().__init__(*args, **kwargs)
        if model_name == 'efficientnet_s':
            weights = EfficientNet_V2_S_Weights.DEFAULT
            model = efficientnet_v2_s(weights=weights)
        elif model_name == 'efficientnet_m':
            weights = EfficientNet_V2_M_Weights.DEFAULT
            model = efficientnet_v2_m(weights=weights)
        else:
            raise NotImplementedError(model_name)

        model.classifier = torch.nn.Identity()
        self.model = model
        self.make_transforms(weights.transforms())
        
        self.output_dim = 1280
    
    def forward(self, img):
        return self.model(img)
    
    def make_transforms(self, transforms):
        target_image_size = transforms.resize_size[0]
        self.eval_transforms = get_eval_transforms(transforms.mean, transforms.std, target_img_size=target_image_size)
        self.train_transforms = get_train_transforms(transforms.mean, transforms.std, target_img_size=target_image_size)
        

def get_encoder(model_name='uni_v1', cache_dir='timm_cache', target_img_size=224):
    #print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name in ('efficientnet_s', 'efficientnet_m'):
        model = TorchVisionEfficientNet(model_name)
        return model, model.eval_transforms, model.train_transforms, model.output_dim
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'dino_v2':
        model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, cache_dir=cache_dir)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'conch_v1_5':
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Please install huggingface transformers (e.g. 'pip install transformers') to use CONCH v1.5")
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, _ = titan.return_conch()
        assert target_img_size == 448, 'TITAN is used with 448x448 CONCH v1.5 features'
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    #print(model)
    constants = MODEL2CONSTANTS[model_name]
    eval_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)
    train_transforms = get_train_transforms(mean=constants['mean'],
                                            std=constants['std'],
                                            target_img_size = target_img_size)
    output_dim = constants['output_dim']
    return model, eval_transforms, train_transforms, output_dim




def get_transforms(model_name, target_img_size=224):
    constants = MODEL2CONSTANTS[model_name]
    eval_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)
    train_transforms = get_train_transforms(mean=constants['mean'],
                                            std=constants['std'],
                                            target_img_size = target_img_size)
    return eval_transforms, train_transforms
    

def get_eval_transforms(mean, std, target_img_size = -1):
    trsforms = []
	#trsforms.append(transforms.ToTensor())  # DomainShift: We're using torch tensors, not PIL images
    if target_img_size > 0:
        trsforms.append(transforms.Resize(size=(target_img_size, target_img_size)))

    trsforms.append(transforms.ToDtype(torch.float32, scale=True))
    trsforms.append(transforms.Normalize(mean, std))
    trsforms = torch.nn.Sequential(*trsforms)  # Switched from transforms.Compose as suggest by torchvision docs

    return trsforms


def get_train_transforms(mean, std, target_img_size = -1):
    trsforms = []
	#trsforms.append(transforms.ToTensor())  # DomainShift: We're using torch tensors, not PIL images
    trsforms.append(transforms.RandAugment())
	
    if target_img_size > 0:
        trsforms.append(transforms.RandomResizedCrop(size=(target_img_size, target_img_size), antialias=True))
	
    trsforms.append(transforms.ToDtype(torch.float32, scale=True))
    trsforms.append(transforms.Normalize(mean, std))
    trsforms = torch.nn.Sequential(*trsforms)  # Switched from transforms.Compose as suggest by torchvision docs

    return trsforms


class PredictionHead(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, *args, dropout_rate=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        projection = torch.nn.Linear(input_dim, hidden_dim)
        if n_classes == 2:
            # In the case of binary prediction, we're using one BCE with logits loss
            output_projection = torch.nn.Linear(hidden_dim, 1)
        else:
            output_projection = torch.nn.Linear(hidden_dim, n_classes)
        classification_layers = [projection, torch.nn.ReLU()]
        if dropout_rate > 1e-3:
            classification_layers.append(torch.nn.Dropout(dropout_rate))
        #classification_layers.append(torch.nn.BatchNorm1d(hidden_dim))
        classification_layers.append(output_projection)
        self.classificaton_head = torch.nn.Sequential(*classification_layers)
        
    def forward(self, state_from_below):
        logits = self.classificaton_head(state_from_below)
        return logits
    

class Classifier(lightning.LightningModule):
    def __init__(self, encoder, input_dim, hidden_dim, n_classes, *args, learning_rate=1e-3, dropout_rate=0.5,  **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.encoder = encoder
        self.classificaton_head = PredictionHead(input_dim, hidden_dim, n_classes, dropout_rate=dropout_rate)
        self.learning_rate = learning_rate
        self.loss_fn = F.binary_cross_entropy_with_logits if n_classes == 2 else F.cross_entropy
       
        self.save_hyperparameters()
    
    def unfreeze_encoder(self):
        self.train_encoder = True
        for p in self.encoder.parameters():
            p.requires_grad = True
        
    def predict(self, img):
        img_repr = self.encoder(img)
        logits = self.classificaton_head(img_repr)
        return logits

    def predict_proba(self, img):
        logits = self.predict(img)
        if self.n_classes == 2:
            probs_pos = torch.sigmoid(logits.squeeze(1))
            probs = torch.stack([1.0 - probs_pos, probs_pos], dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
        return probs
    
    def loss_on_batch(self, img, label):
        logits = self.predict(img)
        loss = self.loss_fn(logits.squeeze(1), label)
        return loss
    
    def training_step(self, batch, batch_idx):
        x, label = batch
        loss = self.loss_on_batch(x, label)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, label = batch
        val_loss = self.loss_on_batch(x, label)
        self.log("val_loss", val_loss)
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, label = batch
        loss = self.loss_on_batch(x, label)
        self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def forward(self, images):
        logits = self.predict(images)
        return logits
        
class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module: Classifier):
         self.freeze(pl_module.encoder, train_bn=False)

    def finetune_function(self, pl_module: Classifier, current_epoch, optimizer):
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.encoder,
                optimizer=optimizer,
                train_bn=True,
            )


class MultiTaskClassifier(lightning.LightningModule):
    def __init__(self, encoder, input_dim, hidden_dim, tasks, task_class_mappings, task_weights, *args, learning_rate=1e-3, dropout_rate=0.5,  **kwargs):
        super().__init__(*args, **kwargs)
        self.task_class_mappings = task_class_mappings
        self.encoder = encoder
        self.tasks = tasks
        self.task_weights= [task_weights[t] for t in tasks]
        self.classificaton_heads = torch.nn.ModuleList()
        self.loss_fns = []
        self.n_classes = []
        for task in self.tasks:
            task_class_mapping = self.task_class_mappings[task]
            n_classes = len(task_class_mapping)
            self.n_classes.append(n_classes)
            task_classificaton_head = PredictionHead(input_dim, hidden_dim, n_classes, dropout_rate=dropout_rate)
            self.classificaton_heads.append(task_classificaton_head)
            loss_fn = F.binary_cross_entropy_with_logits if n_classes == 2 else F.cross_entropy
            self.loss_fns.append(loss_fn)
        self.learning_rate = learning_rate
        
        self.save_hyperparameters()
    
    def unfreeze_encoder(self):
        self.train_encoder = True
        for p in self.encoder.parameters():
            p.requires_grad = True
        
    def predict(self, img):
        img_repr = self.encoder(img)
        logits = [classification_head(img_repr) for classification_head in self.classificaton_heads]
        return logits
    
    def loss_on_batch(self, img, labels, masks):
        logits = self.predict(img)
        all_losses = []
        # Everything is ordered by the name of the tasks, we just zip them together to route the correct entities together
        for logit, label, n_classes, weight, mask in zip(logits, labels, self.n_classes, self.task_weights, masks):
            if n_classes == 2:
                unreduced_loss = F.binary_cross_entropy_with_logits(logit.squeeze(1), label.to(logit.dtype), reduction='none')
            else:
                unreduced_loss = F.cross_entropy(logit.squeeze(1), label, reduction='none')
            masked_loss = unreduced_loss*mask
            reduced_loss = torch.sum(masked_loss)/torch.sum(mask)
            loss = weight*reduced_loss
            all_losses.append(loss)
        return torch.sum(torch.stack(all_losses))
    
    def training_step(self, batch, batch_idx):
        x, labels, masks = batch
        loss = self.loss_on_batch(x, labels, masks)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, labels, masks = batch
        val_loss = self.loss_on_batch(x, labels, masks)
        self.log("val_loss", val_loss)
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, labels, masks = batch
        loss = self.loss_on_batch(x, labels, masks)
        self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def forward(self, images):
        logits = self.predict(images)
        return logits
    
    def predict_tasks(self, images):
        logits = self(images)
        task_predictions = {}
        if not hasattr(self, 'inv_class_mappings'):
            self.inv_class_mappings = {}
            for task, class_mapping in self.task_class_mappings.items():
                inv_class_mapping = {v:k for k,v in class_mapping.items()}
                self.inv_class_mappings[task] = inv_class_mapping
        for logit, task in zip(logits, self.tasks):
            task_class_mapping = self.task_class_mappings[task]
            if len(task_class_mapping) == 2:
                probs = torch.sigmoid(logit.squeeze(1))
                predictions = (probs > 0.5).long()
            else:
                predictions = torch.argmax(logit, dim=-1)
            predictions = [self.inv_class_mappings[task][pred.item()] for pred in predictions]
            task_predictions[task] = predictions
        return task_predictions