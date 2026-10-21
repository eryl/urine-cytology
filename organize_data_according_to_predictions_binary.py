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
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    images_dir: Path = args.test_image_dir    
    output_dir: Path = images_dir / 'classified'
    output_dir.mkdir(exist_ok=True)
    
    
    rng = random.Random(args.random_seed)
    
    prediction_files = args.predictions_dir.glob('*.csv')
    prediction_dfs = []
    
    for prediction_file in prediction_files:
        prediction_df = pd.read_csv(prediction_file)            
        prediction_dfs.append(prediction_df)
    
    all_predictions = pd.concat(prediction_dfs)
    per_image_predictions = (all_predictions
                             .groupby('image_names', as_index=False)
                             .agg(prob_mean=('probabilities', 'mean'),
                                  prob_std=('probabilities', 'std'))
                            )
    for record in per_image_predictions.to_dict('records'):
        image_name = record['image_names']
        prediction_mean = record['prob_mean']
        prediction_std = record['prob_std']
        if prediction_mean > args.threshold:
            output_dir_class = output_dir / 'positive'
        else:
            output_dir_class = output_dir / 'negative'
            
        src_image_path = images_dir / image_name
        
        output_dir_class.mkdir(exist_ok=True, parents=True)
        output_name = f"{prediction_mean:.04f}({prediction_std:.04f})_{image_name}"
        output_path = output_dir_class / output_name
        output_path.symlink_to(src_image_path.absolute())

if __name__ == '__main__':
    main()
    