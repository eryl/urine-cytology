import argparse
from pathlib import Path

import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_directories", help="Directories with annotated data. Should contain a CSV with labels, and a directory with the images", nargs='+', type=Path)
    parser.add_argument("output_directory", help="Directory to save the data to", type=Path)
    args = parser.parse_args()

    dataframes = []
    images = dict()
    for directory in args.image_directories:
        if not directory.exists():
            raise OSError(f"No such directory: {directory}")
        dir_csvs = set(directory.glob('*.csv'))
        for csv_file in sorted(dir_csvs):
            dataframe = pd.read_csv(csv_file)
            dataframe['source_csv'] = str(csv_file)
            dataframes.append(dataframe)
        dir_images = set(directory.glob('images/*.png'))
        for img in dir_images:
            basename = img.name
            if basename in images:
                raise RuntimeError(f"Multiple images with the same name: {basename} at {img} and {images[basename]}")
            images[basename] = img
        
    combined_dataframe = pd.concat(dataframes, ignore_index=True)
    # Remove invalid images
    combined_dataframe = combined_dataframe.dropna(subset=['nc_ratio', 'quality'], how='all').drop_duplicates()  # Drop frames without either quality or nc_ratio set
    # Only keep images where the image are in the images directory
    with_images = combined_dataframe[combined_dataframe['image'].map(lambda x: x in images.keys())]
    args.output_directory.mkdir(exist_ok=True, parents=True)
    csv_output = args.output_directory / "combined_annotations.csv"
    with_images.to_csv(csv_output, index=False)
    image_dir = args.output_directory / 'images'
    image_dir.mkdir(exist_ok=True, parents=True)
    for image in with_images['image']:
        image_path = images[image]
        target_path = image_dir / image_path.name
        target_path.symlink_to(image_path)

    print(images)

if __name__ == '__main__':
    main()