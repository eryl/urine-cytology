"""
Run every per-target classifier trained by train_per_target_classifier.py on a new image dataset.

Each (task, outer_fold, inner_fold) model found under experiment_dir produces one CSV:
  <experiment_dir>/test_predictions_<test_image_dir name>/<task>/<outer>/<inner>/predictions.csv

with the same columns as the dev/test predictions written during training
(image_name, predicted_label, predicted_class, prob_<label>...).

For speed, each GPU holds --models-per-pass models at once and runs every image batch through all of
them (so images are decoded once per group rather than once per model), with fp16 autocast and
channels_last. fp16 is used over bf16 because it stays much closer to fp32 outputs at the same speed.
Models are also run through torch.compile (disable with --no-compile). Parameters are graph inputs, so
the compiled code is shared by all fold models: one compile per head shape (binary / multiclass) per GPU.
torch.compile needs the Python C headers (Python.h), e.g. `sudo apt install python3.12-dev`.

--inner-folds restricts which inner cross-validation folds are used, e.g. `--inner-folds 0` runs one
model per (task, outer fold) instead of the full outer x inner ensemble.

Usage:
    python test_per_target_multitask_classifier_distributed.py test_image_dir experiment_dir [options]

experiment_dir is the timestamped run directory (or the latest_experiment symlink), i.e. the
directory containing the <task>/<outer>/<inner> folders.
"""

import argparse
import csv
import json
import os
import random
import sysconfig
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_per_target_classifier import NamedImageDataset, build_classifier


def collect_jobs(experiment_dir: Path, predictions_dir: Path, inner_folds=None) -> list:
    """One job per trained fold directory: (fold_dir, checkpoint, output_path)."""
    jobs = []
    for metadata_path in sorted(experiment_dir.glob("*/*/*/metadata.json")):
        fold_dir = metadata_path.parent
        if inner_folds is not None and fold_dir.name not in inner_folds:
            continue
        # Interrupted runs can leave stale checkpoints behind; the newest one is the one training
        # finished with (and used for its dev/test predictions)
        checkpoints = sorted(fold_dir.glob("lightning_logs/*/checkpoints/*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not checkpoints:
            print(f"No checkpoint in {fold_dir}, skipping")
            continue
        if len(checkpoints) > 1:
            print(f"Multiple checkpoints in {fold_dir}, using newest: {checkpoints[-1].name}")
        output_path = predictions_dir / fold_dir.relative_to(experiment_dir) / "predictions.csv"
        jobs.append((fold_dir, checkpoints[-1], output_path))
    return jobs


def load_fold_model(fold_dir: Path, checkpoint: Path, args, device: torch.device):
    with open(fold_dir / "metadata.json") as fp:
        metadata = json.load(fp)
    n_classes = metadata["n_classes"]
    # json.dump turned boolean labels into "true"/"false" keys; restore the True/False spelling
    # the training-time prediction CSVs use
    inv_label_encoding = {
        v: {"true": "True", "false": "False"}.get(k, k) for k, v in metadata["label_encoding"].items()
    }

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]
    classifier, eval_transforms, _ = build_classifier(
        args.model_name, hparams["hidden_dim"], n_classes, learning_rate=hparams["learning_rate"]
    )
    classifier.load_state_dict(ckpt["state_dict"])
    classifier.to(device, memory_format=torch.channels_last)
    classifier.eval()
    if args.compile:
        classifier.predict_proba = torch.compile(classifier.predict_proba, dynamic=False)
    labels = [inv_label_encoding.get(c, str(c)) for c in range(n_classes)]
    return classifier, eval_transforms, labels


def predict_fold_group(group: list, image_names: list, args, device: torch.device, num_workers: int):
    """Run several fold models over the images in a single pass, so each image is decoded once per group."""
    models = [load_fold_model(fold_dir, checkpoint, args, device) for fold_dir, checkpoint, _ in group]

    # eval_transforms is Sequential(Resize, ToDtype, Normalize): resize on CPU and ship uint8 to the GPU,
    # do the float conversion and normalization there. All models share the same encoder type and transforms.
    eval_transforms = models[0][1]
    cpu_transforms, gpu_transforms = eval_transforms[:1], eval_transforms[1:].to(device)
    dataset = NamedImageDataset(args.test_image_dir, image_names, cpu_transforms)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=num_workers,
        shuffle=False, drop_last=False, pin_memory=True,
    )

    files, writers, tmp_paths = [], [], []
    for (_, _, output_path), (_, _, labels) in zip(group, models):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(".tmp")
        f = open(tmp, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["image_name", "predicted_label", "predicted_class"] + [f"prob_{l}" for l in labels])
        files.append(f)
        writers.append(writer)
        tmp_paths.append(tmp)

    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for images, names in tqdm(loader, desc=f"{device} batches", position=args.num_gpus + device.index,
                                      leave=False, dynamic_ncols=True):
                n_images = len(names)
                if n_images < args.batch_size:
                    # Pad the final batch so the compiled models never see a new input shape
                    images = torch.cat([images, images.new_zeros((args.batch_size - n_images, *images.shape[1:]))])
                images = gpu_transforms(images.to(device, non_blocking=True))
                images = images.contiguous(memory_format=torch.channels_last)
                for (classifier, _, labels), writer in zip(models, writers):
                    probs = classifier.predict_proba(images)[:n_images].float()
                    pred_classes = probs.argmax(dim=1).tolist()
                    probs = probs.cpu().tolist()
                    writer.writerows(
                        [name, labels[pred_cls], pred_cls] + [round(p, 6) for p in row_probs]
                        for name, pred_cls, row_probs in zip(names, pred_classes, probs)
                    )
    finally:
        for f in files:
            f.close()

    for tmp, (_, _, output_path) in zip(tmp_paths, group):
        tmp.rename(output_path)


def run_on_gpu(rank: int, world_size: int, args, job_shards: list, image_names: list):
    """Worker process: owns one GPU and runs its assigned fold models over all images."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    num_workers = max(1, args.num_workers // world_size)

    jobs = [job for job in job_shards[rank] if not job[2].exists()]  # resume-friendly
    groups = [jobs[i:i + args.models_per_pass] for i in range(0, len(jobs), args.models_per_pass)]
    bar = tqdm(total=len(jobs), desc=f"GPU {rank} models", position=rank, leave=True, dynamic_ncols=True)
    for group in groups:
        bar.set_postfix_str(", ".join(str(fold_dir.relative_to(args.experiment_dir)) for fold_dir, _, _ in group))
        predict_fold_group(group, image_names, args, device, num_workers)
        bar.update(len(group))


def has_python_headers() -> bool:
    include_dirs = [sysconfig.get_paths()["include"]]
    for var in ("C_INCLUDE_PATH", "CPATH"):
        include_dirs += [d for d in os.environ.get(var, "").split(os.pathsep) if d]
    return any((Path(d) / "Python.h").exists() for d in include_dirs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_image_dir", type=Path)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to <experiment_dir>/test_predictions_<test_image_dir name>")
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)  # measured fastest on L4
    parser.add_argument("--models-per-pass", type=int, default=8,
                        help="Fold models held on each GPU at once; every image batch is run through all of them")
    parser.add_argument("--model-name",
                        choices=("uni_v1", "resnet50_trunc", "efficientnet_s", "efficientnet_m"),
                        default="efficientnet_m",
                        help="Must match the --model-name used for training")
    parser.add_argument("--inner-folds", nargs="+", default=None,
                        help="Only use these inner folds (directory names, e.g. 0 1). Default: all")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="torch.compile the models (about 10%% faster on L4)")
    parser.add_argument("--random-seed", type=int, default=1729)
    args = parser.parse_args()

    if args.compile and not has_python_headers():
        raise RuntimeError(
            "torch.compile needs Python.h to build Triton's helpers, and it was not found. Install the headers "
            "(e.g. `sudo apt install python3.12-dev`), point C_INCLUDE_PATH at a directory containing "
            "Python.h, or run with --no-compile."
        )

    args.experiment_dir = args.experiment_dir.resolve()
    predictions_dir = args.output_dir or args.experiment_dir / f"test_predictions_{args.test_image_dir.name}"

    image_names = sorted(p.name for p in args.test_image_dir.glob("*.png"))
    if not image_names:
        raise ValueError(f"No .png images found in {args.test_image_dir}")

    jobs = collect_jobs(args.experiment_dir, predictions_dir, args.inner_folds)
    if not jobs:
        raise ValueError(f"No trained folds (metadata.json + checkpoint) found under {args.experiment_dir}"
                         + (f" for inner folds {args.inner_folds}" if args.inner_folds else ""))
    print(f"{len(image_names)} images, {len(jobs)} models -> {predictions_dir}")

    # Shuffle so tasks with different costs are spread evenly, then shard round-robin across GPUs
    random.Random(args.random_seed).shuffle(jobs)
    world_size = min(args.num_gpus, torch.cuda.device_count(), len(jobs))
    if world_size < 1:
        raise RuntimeError("No CUDA devices available")
    args.num_gpus = world_size
    job_shards = [jobs[i::world_size] for i in range(world_size)]

    if world_size > 1:
        mp.spawn(run_on_gpu, args=(world_size, args, job_shards, image_names), nprocs=world_size, join=True)
    else:
        run_on_gpu(0, 1, args, job_shards, image_names)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
