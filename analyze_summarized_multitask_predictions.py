import argparse
from pathlib import Path
import re


import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summarized_predictions_path', type=Path)
    parser.add_argument('output_path', type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.summarized_predictions_path)
    useful_mask = (df['quality'] == 'Useful') & ( df['not_a_cell'] == False) & (df['no_cytoplasma'] == False)
    useful_regions = df[useful_mask]
    image_grouped = useful_regions.groupby(['predictor', 'base_image'])
    nc_ratios = image_grouped['nc_ratio'].value_counts().unstack(fill_value=0)
    chromatic_features = image_grouped[['irregular_chromatin', 'coarse_chromatin', 'hyperchromasia']].sum()
    other_features = image_grouped[['inflammatory', 'reactive_atypia']].sum()
    summary_df = pd.concat([nc_ratios, chromatic_features, other_features], axis=1).reset_index()
    summary_df.to_csv(args.output_path, index=False)



if __name__ == "__main__":
    main()