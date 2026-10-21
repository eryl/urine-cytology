import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import random
import re
import math

import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata', type=Path, nargs='+')
    args = parser.parse_args()
    metadataframes = []
    for metadata_file in args.metadata:
        df = pd.read_csv(metadata_file)
        df['file'] = metadata_file.stem
        metadataframes.append(df)
    metadata = pd.concat(metadataframes, ignore_index=True)
    label_column = metadata['quality']
    label_values = sorted(label_column.dropna().unique())
    valid_df = metadata[label_column.notna()]
    columns_of_interest = ['nc_ratio', 'quality', 'hyperchromasia', 'coarse_chromatin', 'irregular_chromatin', 'is_group', 'reactive_atypia', 'inflammatory', 'not_a_cell']
    n_cols_of_interest = len(columns_of_interest)
    n_rows = int(math.ceil(math.sqrt(n_cols_of_interest)))
    n_cols = n_cols_of_interest // n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12,10))
    for col, ax in zip(columns_of_interest, axes.flatten()):
        sns.histplot(data=valid_df, x=col, ax=ax, hue='file', multiple='dodge', stat='count')
        ax.tick_params(axis='x', labelrotation=45)

    plt.tight_layout()
    fig.savefig(f"foo.png", dpi=150)

if __name__ == '__main__':
    main()
    