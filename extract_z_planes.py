import os
from multiprocessing import Pool, cpu_count
import tifffile
from PIL import Image
import numpy as np

def extract_z_planes(args):
    ome_path, output_dir, tile_size, level_i = args
    output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(ome_path))[0])
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(ome_path))[0]
    try:
        with tifffile.TiffFile(ome_path) as tif:
            series = tif.series[0]
            level = series.levels[level_i]
            #data = series.asarray()
            data = level.asarray(out='memmap')
            *z, y, x, c  = data.shape
            n_vertical_tiles = int(np.ceil(y / tile_size))
            n_horizontal_tiles = int(np.ceil(x / tile_size))
            
            for zi in range(z):
                for vertical_tile in range(n_vertical_tiles):
                    tile_start_y = vertical_tile*tile_size
                    tile_end_y = tile_start_y + tile_size
                    for horizontal_tile in range(n_horizontal_tiles):
                        tile_start_x = horizontal_tile*tile_size
                        tile_end_x = tile_start_x + tile_size
                        img_arr = data[zi, tile_start_y: tile_end_y, tile_start_x:tile_end_x]
                        out_path = os.path.join(output_dir, f"{basename}_l{level_i}_z{zi}_y{tile_start_y}-{tile_end_y}_x{tile_start_x}-{tile_end_x}.png")
                        #out_path = os.path.join(output_dir, f"{basename}_z{zi}_y{tile_start_y}-{tile_end_y}_x{tile_start_x}-{tile_end_x}.png")
                        im = Image.fromarray(img_arr)
                        im.save(out_path)
                        #tifffile.imwrite(out_path, img_arr)
    except Exception as e:
        print(f"[ERROR] Failed to process {ome_path}: {e}")

def batch_extract_parallel(input_dir, output_dir, num_workers=None, level=0, tile_size=4096):
    os.makedirs(output_dir, exist_ok=True)

    ome_files = [
        (os.path.join(input_dir, f), output_dir, tile_size, level)
        for f in os.listdir(input_dir)
        if f.endswith(".ndpi") or f.endswith(".ndpi")
    ]

    if not ome_files:
        print("No NDPI files found.")
        return

    print(f"Processing {len(ome_files)} files with {num_workers} workers...")
    if num_workers is not None and num_workers > 1:
        with Pool(processes=num_workers or cpu_count()) as pool:
            pool.map(extract_z_planes, ome_files)
    else:
        for ome_file in ome_files:
            extract_z_planes(ome_file)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Z-planes from OME-TIFFs in parallel.")
    parser.add_argument("input_dir", help="Directory containing .ome.tiff files")
    parser.add_argument("output_dir", help="Directory to save extracted Z-planes")
    parser.add_argument("--level", help="Which layer of the file to use", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: all cores)")

    args = parser.parse_args()
    batch_extract_parallel(args.input_dir, args.output_dir, args.workers, args.level)
