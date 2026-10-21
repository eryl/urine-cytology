import argparse
from pathlib import Path

import geojson
import openslide
import numpy as np
from tqdm.auto import tqdm

def get_patches(feature_collection):
    valid_annotations = [ob["geometry"] for ob in feature_collection["features"] if ob["geometry"].get("isEllipse", False)]
    patch_coordinates = []
    for annotation in valid_annotations:
        coordinates = np.array(annotation["coordinates"][0])
        min_x, min_y = np.min(coordinates, axis=0)
        max_x, max_y = np.max(coordinates, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        patch_coordinates.append((int(round(min_x)), int(round(min_y)), int(round(width)), int(round(height))))
    return patch_coordinates
        

def main():
    parser = argparse.ArgumentParser(description="Extract patches from GeoJSON annotations")
    parser.add_argument('annotations_dir', type=Path, help="Directory containing GEOJSON annotation files")
    parser.add_argument('image_dir', type=Path, help="Directory containing images")
    parser.add_argument('output_dir', type=Path, help="Directory where to save image patches to")
    args = parser.parse_args()
    gejson_files = sorted(args.annotations_dir.glob('*.geojson'))
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    for geojson_file in tqdm(gejson_files, desc="GeoJSON file"):
        with open(geojson_file) as fp:
            objects = geojson.load(fp)
        patches = get_patches(objects)
        stem = geojson_file.stem
        image_file = args.image_dir / f"{stem}.ndpi"
        if not image_file.exists():
            print(f"Can't find image {image_file}, skipping {geojson_file}")

        with openslide.OpenSlide(image_file) as slide:
            for (min_x, min_y, width, height) in tqdm(patches, desc="patches"):
                img = slide.read_region((min_x, min_y), 0, (width, height))
                output_name = f"{image_file.stem}_{(min_x, min_y)}_{(width, height)}.png"
                output_path = args.output_dir / output_name
                img.convert("RGB").save(output_path)


if __name__ == '__main__':
    main()