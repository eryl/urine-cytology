"""
Train one classifier per target variable per cross-validation fold.

Instead of a single multi-head network, each (task, outer_fold, inner_fold) combination
gets its own independent Classifier model.  Jobs are sharded round-robin across GPUs so
all devices stay busy.

After each model finishes training, predictions on the dev and test splits are written to
  <output_dir>/<experiment_name>/per_target/<timestamp>/<task>/<outer>/<inner>/dev_predictions.csv
  <output_dir>/<experiment_name>/per_target/<timestamp>/<task>/<outer>/<inner>/test_predictions.csv

Usage:
    python train_per_target_classifier.py image_dir split_file metadata_file experiment_name [options]
"""

import argparse
import csv
import datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image
from tqdm import tqdm

import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from models import get_encoder, Classifier, FeatureExtractorFreezeUnfreeze


ENCODING_OVERRIDES = {
    'nc_ratio': {'NC ratio < 0.5': 0, 'NC ratio 0.5 < 0.7': 1, 'NC ratio > 0.7': 2},
    'quality': {'Useless': 0, 'Poor quality': 1, 'Useful': 2},
}


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def collate_task_mapping(task_metadata: pd.DataFrame) -> dict:
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
    for task, values in task_classes.items():
        if task in ENCODING_OVERRIDES:
            task_label_encodings[task] = ENCODING_OVERRIDES[task]
        else:
            task_label_encodings[task] = {label: i for i, label in enumerate(sorted(values))}
    return task_label_encodings


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SingleTaskDataset(Dataset):
    """Images labelled for one target; images without a valid label are excluded."""

    def __init__(self, image_dir: Path, image_names, metadata: pd.DataFrame,
                 task: str, label_encoding: dict, transforms=None):
        self.image_dir = image_dir
        self.label_encoding = label_encoding
        self.transforms = transforms
        meta_index = metadata.set_index('image')[task]
        self.samples = []
        for name in image_names:
            if name not in meta_index.index:
                continue
            val = meta_index[name]
            try:
                if math.isnan(val):
                    continue
            except TypeError:
                pass
            if val not in label_encoding:
                continue
            self.samples.append((name, label_encoding[val]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        image = decode_image(str(self.image_dir / name))
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label


class NamedImageDataset(Dataset):
    """Inference-only: returns (image, filename)."""

    def __init__(self, image_dir: Path, image_names: list, transforms=None):
        self.image_dir = image_dir
        self.image_names = image_names
        self.transforms = transforms

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        image = decode_image(str(self.image_dir / name))
        if self.transforms is not None:
            image = self.transforms(image)
        return image, name


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_classifier(model_name: str, hidden_dim: int, n_classes: int, learning_rate: float):
    model, eval_transforms, train_transforms, output_dim = get_encoder(model_name=model_name)
    classifier = Classifier(model, output_dim, hidden_dim, n_classes, learning_rate=learning_rate)
    return classifier, eval_transforms, train_transforms


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_and_save(
    checkpoint_path: Path,
    image_names: list,
    image_dir: Path,
    eval_transforms,
    model_name: str,
    hidden_dim: int,
    n_classes: int,
    inv_label_encoding: dict,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    if not image_names:
        return

    classifier, _, _ = build_classifier(model_name, hidden_dim, n_classes, learning_rate=1e-4)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    classifier.load_state_dict(ckpt['state_dict'])
    classifier.to(device)
    classifier.eval()

    dataset = NamedImageDataset(image_dir, image_names, eval_transforms)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, drop_last=False,
        persistent_workers=num_workers > 0,
    )

    rows = []
    with torch.inference_mode():
        for images, names in loader:
            logits = classifier(images.to(device))
            if n_classes == 2:
                probs_pos = torch.sigmoid(logits.squeeze(1))
                probs = torch.stack([1.0 - probs_pos, probs_pos], dim=1)
            else:
                probs = torch.softmax(logits, dim=1)
            pred_classes = probs.argmax(dim=1)
            for i, name in enumerate(names):
                row = {'image_name': name}
                pred_cls = pred_classes[i].item()
                row['predicted_label'] = inv_label_encoding.get(pred_cls, str(pred_cls))
                row['predicted_class'] = pred_cls
                for c in range(n_classes):
                    lbl = inv_label_encoding.get(c, str(c))
                    row[f'prob_{lbl}'] = round(probs[i, c].item(), 6)
                rows.append(row)

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    tmp = output_path.with_suffix('.tmp')
    with open(tmp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.rename(output_path)


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class EpochProgressCallback(lightning.Callback):
    """Prints one line per validation epoch with train/val loss and early-stop patience."""

    def __init__(self, prefix: str, total_epochs: int, patience: int):
        self.prefix = prefix
        self.total_epochs = total_epochs
        self.patience = patience
        self._best_val = float('inf')
        self._epochs_without_improvement = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch + 1
        train_loss = float(metrics.get('train_loss', float('nan')))
        val_loss = float(metrics.get('val_loss', float('nan')))
        if val_loss < self._best_val:
            self._best_val = val_loss
            self._epochs_without_improvement = 0
            marker = '*'
        else:
            self._epochs_without_improvement += 1
            marker = f'+{self._epochs_without_improvement}/{self.patience}'
        print(
            f"  [{self.prefix}] epoch {epoch}/{self.total_epochs}"
            f"  train={train_loss:.4f}  val={val_loss:.4f}  {marker}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Collate functions (module-level so they are picklable by multiprocessing)
# ---------------------------------------------------------------------------

def _collate_float(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return images, labels


def _collate_long(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return images, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_single_task_fold(
    task: str,
    label_encoding: dict,
    train_names: list,
    dev_names: list,
    test_names: list,
    metadata_df: pd.DataFrame,
    args,
    output_dir: Path,
    device: torch.device,
    num_workers: int,
    prefix: str = '',
):
    n_classes = len(label_encoding)
    inv_label_encoding = {v: k for k, v in label_encoding.items()}

    classifier, eval_transforms, train_transforms = build_classifier(
        args.model_name, args.hidden_dim, n_classes, args.learning_rate
    )

    train_ds = SingleTaskDataset(args.image_dir, train_names, metadata_df, task, label_encoding, train_transforms)
    dev_ds   = SingleTaskDataset(args.image_dir, dev_names,   metadata_df, task, label_encoding, eval_transforms)
    test_ds  = SingleTaskDataset(args.image_dir, test_names,  metadata_df, task, label_encoding, eval_transforms)

    if len(train_ds) == 0:
        print(f"[{task}] No training samples, skipping")
        return
    if len(dev_ds) == 0:
        print(f"[{task}] No dev samples, skipping")
        return

    # binary targets must be float; multiclass must be long
    collate = _collate_float if n_classes == 2 else _collate_long

    accumulate = max(1, args.optimization_batch_size // args.device_batch_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.device_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate, persistent_workers=num_workers > 0,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.device_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate,
        persistent_workers=num_workers > 0,
    )

    patience = 20
    total_epochs = args.frozen_epochs + args.unfrozen_epochs
    ckpt_cb      = ModelCheckpoint(monitor="val_loss", save_top_k=1)
    logger       = TensorBoardLogger(save_dir=str(output_dir), version=1, name="lightning_logs")
    early_stop   = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    unfreeze_cb  = FeatureExtractorFreezeUnfreeze(args.frozen_epochs)
    progress_cb  = EpochProgressCallback(prefix=prefix or task, total_epochs=total_epochs, patience=patience)

    trainer = lightning.Trainer(
        accelerator='gpu' if device.type == 'cuda' else 'cpu',
        devices=[device.index] if device.type == 'cuda' else 1,
        callbacks=[unfreeze_cb, ckpt_cb, early_stop, progress_cb],
        logger=logger,
        max_epochs=total_epochs,
        precision="bf16-mixed",
        log_every_n_steps=4,
        accumulate_grad_batches=accumulate,
        enable_progress_bar=False,
    )

    with open(output_dir / 'metadata.json', 'w') as fp:
        json.dump({
            'task': task,
            'n_classes': n_classes,
            'label_encoding': label_encoding,
            'train_images': list(train_names),
            'dev_images': list(dev_names),
            'test_images': list(test_names),
        }, fp, indent=2)

    trainer.fit(classifier, train_loader, dev_loader)

    best_ckpt = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else None
    if best_ckpt is None or not best_ckpt.exists():
        candidates = sorted((output_dir / "lightning_logs").glob("**/checkpoints/*.ckpt"))
        if not candidates:
            print(f"[{task}] No checkpoint found, skipping predictions")
            return
        best_ckpt = candidates[-1]

    infer_bs = args.device_batch_size * 4
    predict_and_save(
        best_ckpt, list(dev_names), args.image_dir, eval_transforms,
        args.model_name, args.hidden_dim, n_classes, inv_label_encoding,
        output_dir / 'dev_predictions.csv', device, infer_bs, num_workers,
    )
    predict_and_save(
        best_ckpt, list(test_names), args.image_dir, eval_transforms,
        args.model_name, args.hidden_dim, n_classes, inv_label_encoding,
        output_dir / 'test_predictions.csv', device, infer_bs, num_workers,
    )


# ---------------------------------------------------------------------------
# GPU worker (one process per GPU)
# ---------------------------------------------------------------------------

def run_on_gpu(rank: int, world_size: int, args, job_shards: list, metadata_df: pd.DataFrame):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    num_workers = max(1, args.num_workers // world_size)
    jobs = job_shards[rank]

    bar = tqdm(jobs, desc=f"GPU {rank}", position=rank, leave=True, dynamic_ncols=True)
    for job in bar:
        task, outer_fold, inner_fold, label_encoding, train_names, dev_names, test_names, output_dir_str = job
        output_dir = Path(output_dir_str)
        prefix = f"GPU{rank} {task} {outer_fold}/{inner_fold}"
        bar.set_postfix_str(f"{task} {outer_fold}/{inner_fold}")

        if (output_dir / 'test_predictions.csv').exists():
            tqdm.write(f"[{prefix}] skip (done)")
            continue

        tqdm.write(f"[{prefix}] starting")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            train_single_task_fold(
                task=task, label_encoding=label_encoding,
                train_names=train_names, dev_names=dev_names, test_names=test_names,
                metadata_df=metadata_df, args=args,
                output_dir=output_dir, device=device, num_workers=num_workers,
                prefix=prefix,
            )
        except Exception as exc:
            tqdm.write(f"[{prefix}] ERROR: {exc}")
            raise


# ---------------------------------------------------------------------------
# Job collection
# ---------------------------------------------------------------------------

def collect_jobs(split_definitions: dict, task_class_mappings: dict, experiment_root: Path) -> list:
    jobs = []
    for outer_fold, inner_folds in split_definitions['splits'].items():
        for inner_fold, splits in inner_folds.items():
            train_names = [img for img, _ in splits['train']]
            dev_names   = [img for img, _ in splits['dev']]
            test_names  = [img for img, _ in splits['test']]
            for task, label_encoding in task_class_mappings.items():
                output_dir = experiment_root / task / outer_fold / inner_fold
                jobs.append((
                    task, outer_fold, inner_fold, label_encoding,
                    train_names, dev_names, test_names, str(output_dir),
                ))
    return jobs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image_dir', type=Path)
    parser.add_argument('split_file', type=Path)
    parser.add_argument('metadata_file', type=Path)
    parser.add_argument('experiment_name', help="Experiment name (used as output subdirectory)")
    parser.add_argument('--output-dir', type=Path, default=Path('experiments'))
    parser.add_argument('--resume-from', type=Path, default=None, help="Resume from checkpoint (optional)")
    parser.add_argument('--random-seed', type=int, default=1729)
    parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--num-workers', type=int, default=14)
    parser.add_argument('--optimization-batch-size', type=int, default=128)
    parser.add_argument('--device-batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--frozen-epochs', type=int, default=20)
    parser.add_argument('--unfrozen-epochs', type=int, default=1000)
    parser.add_argument('--model-name',
                        choices=('uni_v1', 'resnet50_trunc', 'efficientnet_s', 'efficientnet_m'),
                        default='efficientnet_m')
    parser.add_argument('--tasks', nargs='+', default=[
        "quality", "nc_ratio", "hyperchromasia", "coarse_chromatin",
        "irregular_chromatin", "reactive_atypia", "is_group",
        "inflammatory", "not_a_cell", "no_cytoplasma",
    ])
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)

    with open(args.split_file) as fp:
        split_definitions = json.load(fp)

    metadata = pd.read_csv(args.metadata_file)

    for task in args.tasks:
        if task not in metadata.columns:
            raise ValueError(f"Task '{task}' not found in metadata columns: {list(metadata.columns)}")

    task_class_mappings = collate_task_mapping(metadata[args.tasks])

    if args.resume_from is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        experiment_class_dir = args.output_dir / args.experiment_name / "per_target"
        experiment_root = experiment_class_dir / timestamp
        experiment_root.mkdir(parents=True, exist_ok=False)
    else:
        experiment_root = args.resume_from
        if not experiment_root.exists():
            raise ValueError(f"Resume-from path does not exist: {experiment_root}")
        experiment_class_dir = experiment_root.parent

    latest = experiment_class_dir / 'latest_experiment'
    if latest.exists():
        latest.unlink()
    latest.symlink_to(experiment_root.relative_to(experiment_class_dir))

    jobs = collect_jobs(split_definitions, task_class_mappings, experiment_root)
    indices = rng.permutation(len(jobs))
    jobs = [jobs[i] for i in indices]
    print(f"Total jobs: {len(jobs)}  ({len(args.tasks)} tasks × folds)")

    n_cuda = torch.cuda.device_count()
    world_size = min(args.num_gpus, n_cuda, len(jobs))
    job_shards = [jobs[i::world_size] for i in range(world_size)]

    if world_size > 1:
        mp.spawn(run_on_gpu, args=(world_size, args, job_shards, metadata),
                 nprocs=world_size, join=True)
    elif world_size == 1:
        run_on_gpu(0, 1, args, job_shards, metadata)
    else:
        # No GPUs — run on CPU
        device = torch.device('cpu')
        bar = tqdm(jobs, desc="CPU", leave=True, dynamic_ncols=True)
        for job in bar:
            task, outer_fold, inner_fold, label_encoding, train_names, dev_names, test_names, output_dir_str = job
            output_dir = Path(output_dir_str)
            prefix = f"{task} {outer_fold}/{inner_fold}"
            bar.set_postfix_str(prefix)
            if (output_dir / 'test_predictions.csv').exists():
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            train_single_task_fold(
                task=task, label_encoding=label_encoding,
                train_names=train_names, dev_names=dev_names, test_names=test_names,
                metadata_df=metadata, args=args, output_dir=output_dir,
                device=device, num_workers=args.num_workers,
                prefix=prefix,
            )


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
