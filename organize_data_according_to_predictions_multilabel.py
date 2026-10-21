import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Set, Tuple
import datetime
import shutil

import numpy as np
import pandas as pd
from tqdm import trange, tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_image_dir', type=Path)
    parser.add_argument('predictions_dir', type=Path)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--frozen-epochs', type=int, default=20)
    parser.add_argument('--unfrozen-epochs', type=int, default=1000)
    parser.add_argument('--random-seed', type=int, default=1729)
    parser.add_argument('--model-name', choices=('uni_v1', 'resnet50_trunc'), default='resnet50_trunc')
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--sample-size', type=int, default=4000)
    
    args = parser.parse_args()
    images_dir: Path = args.test_image_dir    
    output_dir: Path = images_dir / f'MI_sorted_{args.sample_size}'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    rng = random.Random(args.random_seed)
    
    prediction_files = args.predictions_dir.glob('*.csv')
    prediction_dfs = []
    
    for prediction_file in prediction_files:
        prediction_df = pd.read_csv(prediction_file)            
        prediction_dfs.append(prediction_df)
    
    all_predictions = pd.concat(prediction_dfs)
    prob_cols = [c for c in all_predictions.columns if c.startswith("probabilities_")]
    
    eps = 1e-12  # numerical stability
    # (Optional) make sure each row is a valid distribution
    probs = all_predictions[prob_cols].clip(lower=0, upper=1)
    probs = probs.div(probs.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0/len(prob_cols))
    all_predictions[prob_cols] = probs

    # Per-row (per-model) entropy H(p_i)
    row_entropy = -(probs.clip(lower=eps).apply(np.log, axis=1) * probs).sum(axis=1)
    all_predictions['row_entropy'] = row_entropy

    # Mean probabilities per image (your aggregation)
    per_image = (all_predictions
                .groupby('image_names', as_index=False)[prob_cols]
                .mean())

    # Predictive entropy H(mean_p)
    mean_probs = per_image[prob_cols]
    pred_entropy = -(mean_probs.clip(lower=eps).apply(np.log, axis=1) * mean_probs).sum(axis=1)

    # Mean per-model entropy E[H(p_i)] per image
    mean_model_entropy = (all_predictions
                        .groupby('image_names')['row_entropy']
                        .mean()
                        .reindex(per_image['image_names'].values)
                        .values)

    # Mutual Information = H(mean_p) - E[H(p_i)]
    per_image['mutual_information'] = pred_entropy - mean_model_entropy

    # Margin = top1(mean_p) - top2(mean_p)
    sorted_mean = np.sort(mean_probs.values, axis=1)[:, ::-1]
    per_image['margin'] = sorted_mean[:, 0] - sorted_mean[:, 1]

    mi_sorter = np.argsort(per_image['mutual_information'])
    selected_images = per_image.iloc[mi_sorter][-args.sample_size:]
    
    for record in selected_images.to_dict('records'):
        image_name = record['image_names']
        src_image_path: Path = images_dir / image_name
        linked_image_path = src_image_path.resolve()
        mi = record['mutual_information']
        output_path: Path = output_dir / f"{mi:.02f}_{image_name}"
        output_path.symlink_to(linked_image_path.absolute())

if __name__ == '__main__':
    main()
    