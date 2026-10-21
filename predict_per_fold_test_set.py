"""
Evaluate each fold's model on its own held-out test images and save predictions.

For every (outer, inner) fold in the experiment directory, reads metadata.json
to find the test images for that fold, loads the corresponding checkpoint, and
writes predictions_{outer}_{inner}.csv to <experiment_dir>/test_predictions_per_fold/.

Usage:
    python predict_per_fold_test_set.py <image_dir> <experiment_dir> [options]
"""

import csv
import json
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image
from tqdm import tqdm

from models import MultiTaskClassifier, get_transforms


class ImageDataset(Dataset):
    def __init__(self, images, transforms=None):
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = decode_image(str(image_path))
        if self.transforms is not None:
            image = self.transforms(image)
        return image, image_path.name


def run_on_gpu(rank: int, world_size: int, args, fold_shards: list):
    """Worker process: owns one GPU, processes its assigned (checkpoint, test_images) pairs."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    eval_transforms, _ = get_transforms(args.model_name)

    predictions_dir = args.experiment_dir / "test_predictions_per_fold"
    predictions_dir.mkdir(exist_ok=True, parents=True)

    for checkpoint, test_image_paths, outer_fold, inner_fold in tqdm(
        fold_shards[rank], desc=f"GPU {rank} folds", position=rank
    ):
        out_path = predictions_dir / f"predictions_{outer_fold}_{inner_fold}.csv"
        if out_path.exists():
            continue

        if not test_image_paths:
            continue

        dataset = ImageDataset(test_image_paths, transforms=eval_transforms)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers // world_size,
            shuffle=False,
            drop_last=False,
            persistent_workers=(args.num_workers // world_size) > 0,
        )

        model = MultiTaskClassifier.load_from_checkpoint(checkpoint)
        model.to(device)
        model.eval()

        tmp_out_path = out_path.with_suffix(".tmp")
        with torch.inference_mode(), open(tmp_out_path, "w", newline="") as csvfile:
            writer = None
            for images, batch_names in tqdm(
                loader, desc="Batches", position=world_size + rank, leave=False
            ):
                task_predictions = model.predict_tasks(images.to(device))
                if writer is None:
                    field_names = ["image_name"] + list(task_predictions.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=field_names)
                    writer.writeheader()
                for i, name in enumerate(batch_names):
                    writer.writerow(
                        {"image_name": name, **{t: task_predictions[t][i] for t in task_predictions}}
                    )
            csvfile.flush()
        tmp_out_path.rename(out_path)


def collect_folds(experiment_dir: Path, image_dir: Path) -> list:
    """Return list of (checkpoint, test_image_paths, outer, inner) for every fold."""
    # Build a name->path index once so we don't glob per fold
    all_images = {p.name: p for p in image_dir.glob("*.png")}

    folds = []
    for outer in sorted(experiment_dir.iterdir()):
        if not outer.is_dir() or not outer.name.isdigit():
            continue
        for inner in sorted(outer.iterdir()):
            if not inner.is_dir() or not inner.name.isdigit():
                continue

            metadata_path = inner / "metadata.json"
            if not metadata_path.exists():
                continue
            with open(metadata_path) as f:
                metadata = json.load(f)

            test_names = metadata.get("test_images", [])
            test_image_paths = [all_images[n] for n in test_names if n in all_images]
            missing = [n for n in test_names if n not in all_images]
            if missing:
                print(f"Warning: {len(missing)} test images missing for fold {outer.name}/{inner.name}")

            checkpoints = sorted((inner / "lightning_logs").glob("**/checkpoints/*.ckpt"))
            if not checkpoints:
                print(f"Warning: no checkpoint found for fold {outer.name}/{inner.name}, skipping")
                continue
            # Take the last checkpoint (sorted lexicographically, last = highest epoch)
            checkpoint = checkpoints[-1]

            folds.append((checkpoint, test_image_paths, outer.name, inner.name))

    return folds


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="Directory containing all cell image PNGs")
    parser.add_argument("experiment_dir", type=Path, help="Root experiment directory (contains fold subdirs)")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--model-name", choices=("uni_v1", "resnet50_trunc"), default="resnet50_trunc")
    args = parser.parse_args()

    folds = collect_folds(args.experiment_dir, args.image_dir)
    print(f"Found {len(folds)} folds to evaluate")

    world_size = min(args.num_gpus, len(folds))

    # Shard folds round-robin across GPUs
    fold_shards = [folds[i::world_size] for i in range(world_size)]

    if world_size > 1:
        mp.spawn(
            run_on_gpu,
            args=(world_size, args, fold_shards),
            nprocs=world_size,
            join=True,
        )
    else:
        run_on_gpu(0, 1, args, fold_shards)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
