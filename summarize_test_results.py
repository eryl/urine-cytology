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


def extract_model_family(model_path):
    model_path = Path(model_path)
    # This looks a bit ugly, but the checkpoints are in a trees under the model family root like `outer_fold_index/inner_fold_index/lightning_logs/version_1/checkpoints`, 
    # to get the model root dire, we then have to travarse this backwards
    *_, model_target_name, model_root_dir, outer_fold_id, inner_fold_id, lightning_log_dir, lightning_version_dir, checkpoint_dir, checkpoint_name = model_path.parts
    return f"{model_target_name}_{model_root_dir}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=Path, help="Directory containing the images")
    parser.add_argument('predictions_roots', type=Path, nargs="+", help="Directory to scan for predictions file. By default anything in folders with the prefix \"test_predictions\", unless --prediction-tag is set")
    parser.add_argument('--predictions-prefix', help="Override what predictions directory prefix to look for", default="test_predictions")
    
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
    
    rng = random.Random(args.random_seed)
    
    all_prediction_files = []
    for prediction_roots in sorted(args.predictions_roots):
        prediction_directories = sorted(prediction_roots.glob(f"**/{args.predictions_prefix}*"))
        for prediction_directory in prediction_directories:
            prediction_files = sorted(prediction_directory.glob('*.csv'))
            all_prediction_files.extend(prediction_files)
    print(all_prediction_files)
    prediction_dfs = []
    
    for prediction_file in all_prediction_files:
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

    all_predictions['model_family'] = all_predictions['model_path'].map(extract_model_family)
    # Mean probabilities per image (your aggregation)
    per_image = (all_predictions
                .groupby(['image_names', 'model_family'], as_index=False)[prob_cols]
                .mean())
    
    label_encodings = {'probabilities_0': 'NC_ratio_0.5-0.7', 
                       'probabilities_1': 'NC_ratio_0-0.5', 
                       'probabilities_2': 'NC_ratio_0.7-1'}

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

    #mi_sorter = np.argsort(per_image['mutual_information'])
    #selected_images = per_image.iloc[mi_sorter][-args.sample_size:]
    
    output_root = args.image_dir / 'test_evaluations'
    output_root.mkdir(exist_ok=True, parents=True)
    for record in per_image.to_dict('records'):
        image_name = record['image_names']
        src_image_path: Path = args.image_dir / image_name
        linked_image_path = src_image_path.resolve()
        mi = record['mutual_information']
        model_family = record['model_family']
        probabilities = [record[col_name] for col_name in prob_cols]
        max_col = prob_cols[np.argmax(probabilities)]
        col_label = label_encodings[max_col]
        output_dir = output_root / model_family
        output_dir.mkdir(exist_ok=True)
        output_path: Path = output_dir / f"{image_name}_{col_label}_MI{mi:.02f}{src_image_path.suffix}"
        output_path.symlink_to(linked_image_path.absolute())

if __name__ == '__main__':
    main()
    