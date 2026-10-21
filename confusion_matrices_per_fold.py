"""
Compute and plot per-task confusion matrices for every fold in a nested-CV experiment.

For each predictions_{outer}_{inner}.csv in <predictions_dir>, the corresponding
model's test images are matched against ground-truth labels from <annotations_csv>.
One figure with one confusion-matrix subplot per task is saved to
<predictions_dir>/confusion_matrices/.

Usage:
    python confusion_matrices_per_fold.py <predictions_dir> <annotations_csv> [--experiment-dir DIR]
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def _class_order(class_mapping: dict) -> list[str]:
    """Return class labels in ascending integer-encoding order."""
    return [k for k, _ in sorted(class_mapping.items(), key=lambda x: x[1])]


def _draw_cm(ax, cm: np.ndarray, labels: list[str], title: str) -> None:
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=9, pad=4)
    ticks = range(len(labels))
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(list(ticks))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylabel("True", fontsize=7)
    ax.set_xlabel("Predicted", fontsize=7)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > thresh else "black",
            )


def process_fold(
    pred_csv: Path,
    annotations: pd.DataFrame,
    class_mappings: dict,
    outer: str,
    inner: str,
    output_dir: Path,
) -> None:
    out_path = output_dir / f"confusion_matrix_{outer}_{inner}.png"
    if out_path.exists():
        return

    preds = pd.read_csv(pred_csv, index_col="image_name")
    # Normalise to lowercase so class_mappings (which use lowercase) align with
    # any capitalisation variant in the CSV (e.g. "False" vs "false")
    preds = preds.apply(lambda col: col.astype(str).str.lower())

    tasks = [t for t in sorted(class_mappings.keys()) if t in preds.columns]
    if not tasks:
        return

    ncols = 5
    nrows = (len(tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.8))
    axes_flat = np.array(axes).flatten()
    fig.suptitle(f"Fold {outer}/{inner} — confusion matrices (n={len(preds)})", fontsize=11)

    for ax_idx, task in enumerate(tasks):
        ax = axes_flat[ax_idx]
        mapping = class_mappings[task]
        labels = [l.lower() for l in _class_order(mapping)]

        if task not in annotations.columns:
            ax.set_visible(False)
            continue

        joined = preds[[task]].join(annotations[task].rename("gt"), how="inner")
        joined = joined.dropna(subset=["gt"])
        joined = joined[joined["gt"] != "nan"]

        y_true = joined["gt"]
        y_pred = joined[task]

        # Keep only rows where both values are recognised class labels
        valid = y_true.isin(labels) & y_pred.isin(labels)
        y_true, y_pred = y_true[valid], y_pred[valid]

        if len(y_true) == 0:
            ax.set_visible(False)
            continue

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        _draw_cm(ax, cm, labels, task)

    for ax_idx in range(len(tasks), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_dir", type=Path,
                        help="Directory containing predictions_{outer}_{inner}.csv files")
    parser.add_argument("annotations_csv", type=Path,
                        help="Ground-truth CSV with an 'image' column and one column per task")
    parser.add_argument("--experiment-dir", type=Path, default=None,
                        help="Root experiment directory containing fold subdirs with metadata.json. "
                             "Defaults to the parent of predictions_dir.")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir or args.predictions_dir.parent

    annotations = pd.read_csv(args.annotations_csv, index_col="image")
    annotations = annotations.apply(lambda col: col.astype(str).str.lower())

    output_dir = args.predictions_dir / "confusion_matrices"
    output_dir.mkdir(exist_ok=True)

    pred_csvs = sorted(args.predictions_dir.glob("predictions_*.csv"))
    if not pred_csvs:
        raise SystemExit(f"No prediction CSVs found in {args.predictions_dir}")

    for pred_csv in tqdm(pred_csvs, desc="Folds"):
        m = re.fullmatch(r"predictions_(\d+)_(\d+)\.csv", pred_csv.name)
        if not m:
            continue
        outer, inner = m.group(1), m.group(2)

        metadata_path = experiment_dir / outer / inner / "metadata.json"
        if not metadata_path.exists():
            print(f"Warning: no metadata.json for fold {outer}/{inner}, skipping")
            continue
        with open(metadata_path) as f:
            class_mappings = json.load(f)["class_mappings"]

        process_fold(pred_csv, annotations, class_mappings, outer, inner, output_dir)

    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
