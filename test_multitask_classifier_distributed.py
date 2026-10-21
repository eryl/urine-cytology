import argparse
import csv
import random
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from torchvision.io import decode_image
from torch.utils.data import Dataset
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


def run_on_gpu(rank: int, world_size: int, args, ckpt_shards: list[Path], image_shard: list[Path]):
    """Worker process: owns one GPU, processes its assigned checkpoints over its image shard."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    checkpoints = ckpt_shards[rank]
    eval_transforms, _ = get_transforms(args.model_name)
    dataset = ImageDataset(image_shard, transforms=eval_transforms)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers // world_size,
        shuffle=False,
        drop_last=False,
        persistent_workers=(args.num_workers // world_size) > 0,
    )

    predictions_dir = args.experiment_dir / f"test_predictions_{args.test_image_dir.name}"
    predictions_dir.mkdir(exist_ok=True, parents=True)

    for ckpt_idx, checkpoint in enumerate(tqdm(checkpoints, desc=f"GPU {rank} checkpoints", position=rank)):
        *_, outer_fold, inner_fold, lightning_logs, version, checkpoints, ckpt_name = checkpoint.parts 
        out_path = predictions_dir / f"predictions_{outer_fold}_{inner_fold}.csv"
        if out_path.exists():
            continue  # resume-friendly
        tmp_out_path = out_path.with_suffix(".tmp")
        model = MultiTaskClassifier.load_from_checkpoint(checkpoint)
        model.to(device)
        model.eval()

        with torch.inference_mode(), open(tmp_out_path, "w", newline="") as csvfile:
            writer = None
            for images, batch_names in tqdm(loader, desc="Batches", position=world_size + rank, leave=False):
                task_predictions = model.predict_tasks(images.to(device))
                if writer is None:
                    field_names = ["image_name"] + list(task_predictions.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=field_names)
                    writer.writeheader()
                for i, name in enumerate(batch_names):
                    writer.writerow({"image_name": name, **{t: task_predictions[t][i] for t in task_predictions}})
            csvfile.flush()
        tmp_out_path.rename(out_path)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("test_image_dir", type=Path)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--model-name", choices=("uni_v1", "resnet50_trunc"), default="resnet50_trunc")
    parser.add_argument("--random-seed", type=int, default=1729)
    args = parser.parse_args()

    test_images = sorted(args.test_image_dir.glob("*.png"))
    checkpoints = sorted(args.experiment_dir.glob("**/*.ckpt"))

    rng = random.Random(args.random_seed)
    rng.shuffle(checkpoints)

    world_size = min(args.num_gpus, len(checkpoints))

    # Shard checkpoints round-robin across GPUs so load is balanced
    ckpt_shards = [checkpoints[i::world_size] for i in range(world_size)]

    # Each GPU sees all images; image sharding only makes sense if you want
    # to further parallelize within a checkpoint — keep it simple and full here.
    # If you DO want image sharding too, swap this for:
    #   img_shards = [test_images[i::world_size] for i in range(world_size)]
    #img_shards = [test_images] * world_size

    mp.spawn(
        run_on_gpu,
        args=(world_size, args, ckpt_shards, test_images),
        # mp.spawn passes rank as first arg; we need per-process ckpt/img shards
        # so wrap to unpack the shard lists by rank:
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()