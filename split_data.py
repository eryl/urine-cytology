import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import random
import re

import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=Path)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--label-column', type=str, default='nc_ratio')
    parser.add_argument('--random-seed', type=int, default=1729)
    parser.add_argument('--k', type=int, default=10, help="How many folds to produce")
    args = parser.parse_args()
    
    #rng = random.Random(args.random_seed)    
    rng = np.random.default_rng(args.random_seed)
    
    images = sorted(args.image_dir.glob('*.png'))
    metadata = pd.read_csv(args.metadata)
    label_column = metadata[args.label_column]
    label_values = sorted(label_column.dropna().unique())
    valid_df = metadata[label_column.notna()]
    label_encodings = {label: i for i,label in enumerate(label_values)}
    #label_encodings = {'Useful': 0, 'Useless': 1, 'Poor quality': 0}
    image_names = []
    y = []

    for record in valid_df.to_dict('records'):
        image_name = record['image']
        image_label = record[args.label_column]
        image_names.append(image_name)
        #encoded_label = label_encodings[image_label]
        y.append(image_label)
                
    k = args.k
    
    split_definition = {"random_seed": args.random_seed,
                        "k": k,
                        "label_column": args.label_column,
                        "label_values": label_values,
                        "label_encodings": label_encodings,
                        "splits": dict()}
    
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=rng.integers(2**32-1))
    X = np.arange(len(image_names))
    for fold_i, (modelling_indices, test_indices) in enumerate(splitter.split(X, y)):
        labeled_test_images = [(image_names[i], y[i]) for i in test_indices]
        modelling_y = [y[i] for i in modelling_indices]

        inner_k = args.k
        inner_splitter = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=rng.integers(2**32-1))
        
        split_definition["splits"][fold_i] = dict()
        for fold_j, (training_indices, dev_indices) in enumerate(inner_splitter.split(modelling_indices, modelling_y)):
            labeled_training_images = [(image_names[i], y[i]) for i in training_indices]
            labeled_dev_images = [(image_names[i], y[i]) for i in dev_indices]
            
            split_definition["splits"][fold_i][fold_j] = {"train": labeled_training_images, 
                                                          "dev": labeled_dev_images, 
                                                          "test": labeled_test_images}
            print(f"{fold_i,fold_j}: train:{Counter([y[i] for i in training_indices])}, dev:{Counter([y[i] for i in dev_indices])}, test: {Counter([y[i] for i in test_indices])}")
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    output_file = args.output_dir / f"splits-non-group-{args.k}-{args.random_seed}_{args.label_column}.json"
    with open(output_file, 'w') as fp:
        json.dump(split_definition, fp, indent=2)
    
    
    
if __name__ == '__main__':
    main()
    