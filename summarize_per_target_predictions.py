"""
Summarize the per-target predictions from test_per_target_multitask_classifier_distributed.py
into per-image feature counts, in the same format as summarize_multitask_predictions.py.

The per-target predictions are laid out as
  <predictions_dir>/<task>/<outer>/<inner>/predictions.csv
with one file per task. The per-task predictions of the same (outer, inner) fold are joined into
one multitask-style predictor named predictions_<outer>_<inner> (matching the file stems of the
multitask predictions), which is then summarized per base image.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import multiprocessing

import pandas as pd
from tqdm import tqdm

REQUIRED_TASKS = ['quality', 'not_a_cell', 'no_cytoplasma', 'nc_ratio', 'irregular_chromatin',
                  'coarse_chromatin', 'hyperchromasia', 'inflammatory', 'reactive_atypia']


def get_base_image(image_name):
    # Implement the logic to extract the base image name
    base_image = image_name.split('_')[0]
    return base_image


def collect_predictors(predictions_dir):
    """Group the per-task prediction files by fold: {(outer, inner): {task: path}}."""
    predictors = defaultdict(dict)
    for prediction_file in predictions_dir.glob('*/*/*/predictions.csv'):
        inner_dir = prediction_file.parent
        outer_dir = inner_dir.parent
        task = outer_dir.parent.name
        predictors[(outer_dir.name, inner_dir.name)][task] = prediction_file
    return predictors


def load_predictor(task_files):
    """Join the predicted labels of all tasks into one dataframe with one column per task."""
    task_dfs = []
    for task, prediction_file in sorted(task_files.items()):
        df = pd.read_csv(prediction_file, usecols=['image_name', 'predicted_label'])
        task_dfs.append(df.set_index('image_name')['predicted_label'].rename(task))
    return pd.concat(task_dfs, axis=1, join='inner').reset_index()


def aggregate_predictors(fold_and_task_files):
    (outer_fold, inner_fold), task_files = fold_and_task_files
    df = load_predictor(task_files)
    df['base_image'] = df['image_name'].map(get_base_image)
    df['predictor'] = f"predictions_{outer_fold}_{inner_fold}"
    useful_mask = (df['quality'] == 'Useful') & ( df['not_a_cell'] == False) & (df['no_cytoplasma'] == False)
    useful_regions = df[useful_mask]
    image_grouped = useful_regions.groupby(['predictor', 'base_image'])
    nc_ratios = image_grouped['nc_ratio'].value_counts().unstack(fill_value=0)
    chromatic_features = image_grouped[['irregular_chromatin', 'coarse_chromatin', 'hyperchromasia']].sum()
    other_features = image_grouped[['inflammatory', 'reactive_atypia']].sum()
    summary_df = pd.concat([nc_ratios, chromatic_features, other_features], axis=1).reset_index()
    return summary_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions_dir', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--workers', help="Number of worker processes", type=int, default=1)
    args = parser.parse_args()

    predictors = collect_predictors(args.predictions_dir)
    if not predictors:
        raise ValueError(f"No <task>/<outer>/<inner>/predictions.csv files found under {args.predictions_dir}")

    jobs = []
    for fold, task_files in sorted(predictors.items()):
        missing = [task for task in REQUIRED_TASKS if task not in task_files]
        if missing:
            print(f"Fold {fold} is missing predictions for {missing}, skipping")
            continue
        jobs.append((fold, task_files))

    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            summarized_dfs = list(tqdm(pool.imap_unordered(aggregate_predictors, jobs), total=len(jobs)))
    else:
        summarized_dfs = [aggregate_predictors(job) for job in tqdm(jobs)]

    all_summarized = pd.concat(summarized_dfs, ignore_index=True).fillna(0)
    all_summarized.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
