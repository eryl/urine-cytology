import argparse
from pathlib import Path
import re
import multiprocessing

import pandas as pd
from tqdm import tqdm

def get_base_image(image_name):
    # Implement the logic to extract the base image name
    base_image = image_name.split('_')[0]
    return base_image

def aggregate_predictors(prediction_file):
    df = pd.read_csv(prediction_file)
    df['base_image'] = df['image_name'].map(get_base_image)
    df['predictor'] = prediction_file.stem
    useful_mask = (df['quality'] == 'Useful') & ( df['not_a_cell'] == False) & (df['no_cytoplasma'] == False)
    #useful_mask = (df['quality'] == 'Useful') & ( df['not_a_cell'] == False)
    useful_regions = df[useful_mask]
    image_grouped = useful_regions.groupby(['predictor', 'base_image'])
    nc_ratios = image_grouped['nc_ratio'].value_counts().unstack(fill_value=0)
    chromatic_features = image_grouped[['irregular_chromatin', 'coarse_chromatin', 'hyperchromasia']].sum()
    other_features = image_grouped[['inflammatory', 'reactive_atypia']].sum()
    summary_df = pd.concat([nc_ratios, chromatic_features, other_features], axis=1).reset_index()
    return summary_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions_dir', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--workers', help="Number of worker processes", type=int)
    args = parser.parse_args()
    
    prediction_files = list(args.predictions_dir.glob('*.csv'))

    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            summarized_dfs = []
            for summarized_df in tqdm(pool.imap_unordered(aggregate_predictors, prediction_files), total=len(prediction_files)):
                summarized_dfs.append(summarized_df)
    else:
        summarized_dfs = tqdm([aggregate_predictors(prediction_file) for prediction_file in prediction_files])
    
    all_summarized = pd.concat(summarized_dfs).reset_index().fillna(0)
    all_summarized.drop('index',axis='columns')
    all_summarized.to_csv(args.output_path, index=False)
    



if __name__ == "__main__":
    main()