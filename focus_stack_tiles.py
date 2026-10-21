import os
import re
import shutil
import tempfile
from pathlib import Path
from glob import glob
from collections import defaultdict
import multiprocessing
import subprocess
from tqdm import tqdm

def extract_tile_key(filename):
    match = re.search(r'(Nr ?\d+)_l\d_z\d+_y(\d+-\d+)_x(\d+-\d+)\.png$', filename)
    if match:
        image_name, y_tile, x_tile = match.groups()
        return image_name, y_tile, x_tile
    return None

def group_tile_stacks(tile_dir):
    tile_groups = defaultdict(lambda: defaultdict(list))
    for file in glob(os.path.join(tile_dir, '*.png')):
        try:
            image_name, y_tile, x_tile = extract_tile_key(file)
            tile_groups[image_name][(y_tile, x_tile)].append(file)
        except ValueError:
            continue
    return tile_groups

def focus_stack_tile(tile_files, output_path):
    # Sort files by Z index
    tile_files = sorted(tile_files, key=lambda f: int(re.search(r'_z(\d+)_', f).group(1)))

    cmd = ['focus-stack',
           '--no-whitebalance',
           '--threads=4',
           #'--no-opencl',
           f'--output={output_path}',
           *tile_files]
    print(" ".join(cmd))
    try:
        p = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in focus-stack for {output_path}: {e}")

def process_work_package(work_package):
    focus_stack_tile(*work_package)

def process_all_tiles(tile_dir, output_dir, num_workers=1):
    os.makedirs(output_dir, exist_ok=True)
    tile_groups = group_tile_stacks(tile_dir)
    print(f"Found {len(tile_groups)} tile stacks to process.")
    work_packages = []
    
    for image_file, image_tiles in tile_groups.items():
            for (y_tile, x_tile), files in image_tiles.items():
                output_path = os.path.join(output_dir, f"{image_file}_y{y_tile}_x{x_tile}_fused.png")
                if not os.path.exists(output_path):
                    work_package = (files, output_path)
                    work_packages.append(work_package)
                else:
                    print(f"Path already processed: {output_path}")

    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            for result in tqdm(pool.imap_unordered(process_work_package, work_packages), total=len(work_packages)):
                continue
    else:       
        for work_package in tqdm(work_packages):
            process_work_package(work_package)
            
# ------------------ Main ------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Focus stack PNG z-tiles in parallel.")
    parser.add_argument("tile_dir", help="Directory with input PNG z-stack tiles")
    parser.add_argument("output_dir", help="Directory to save fused PNG tiles")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel threads")

    args = parser.parse_args()
    process_all_tiles(args.tile_dir, args.output_dir, num_workers=args.workers)
