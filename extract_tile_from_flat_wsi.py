import os
from multiprocessing import Process, Queue, Event
#from multiprocessing.dummy import Process, Queue, Event
import queue
from functools import partial

import openslide
from PIL import Image
import numpy as np
from tqdm import tqdm, trange


def extract_z_planes(args, pbar_pos=0):
    ome_path, output_dir, tile_size, level_i = args
    basename = os.path.splitext(os.path.basename(ome_path))[0]
    output_dir = os.path.join(output_dir, basename)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with openslide.OpenSlide(ome_path) as wsi:
            width, height = wsi.level_dimensions[level_i]
            n_vertical_tiles = int(np.ceil(height / tile_size))
            n_horizontal_tiles = int(np.ceil(width / tile_size))
            
            n_total_tiles = n_horizontal_tiles*n_vertical_tiles
            with tqdm(position=pbar_pos, total=n_total_tiles) as pbar:
                for vertical_tile in range(n_vertical_tiles):
                    tile_start_y = vertical_tile*tile_size
                    tile_end_y = tile_start_y + tile_size
                    for horizontal_tile in range(n_horizontal_tiles):
                        tile_start_x = horizontal_tile*tile_size
                        tile_end_x = tile_start_x + tile_size
                        out_path = os.path.join(output_dir, f"{basename}_l{level_i}_zNone_y{tile_start_y}-{tile_end_y}_x{tile_start_x}-{tile_end_x}.png")
                        if not os.path.exists(out_path):
                            img = wsi.read_region((tile_start_x, tile_start_y), level_i, (tile_size, tile_size)).convert('RGB')
                            img.save(out_path)
                        pbar.update()
    except Exception as e:
        print(f"[ERROR] Failed to process {ome_path}: {e}")
    return ome_path

def worker(pbar_pos, work_queue, results_queue, quit_flag):
    while not quit_flag.is_set():
        try:
            work_package = work_queue.get()
            ome_path = extract_z_planes(work_package, pbar_pos)
            results_queue.put(ome_path)
        except queue.Empty:
            break

def batch_extract_parallel(input_dir, output_dir, num_workers=None, level=0, tile_size=4096):
    os.makedirs(output_dir, exist_ok=True)

    work_packages = [
        (os.path.join(input_dir, f), output_dir, tile_size, level)
        for f in os.listdir(input_dir)
        if f.endswith(".ndpi") or f.endswith(".ndpi")
    ]

    if not work_packages:
        print("No NDPI files found.")
        return

    print(f"Processing {len(work_packages)} files with {num_workers} workers...")
    if num_workers is not None and num_workers > 1:
        processes = []
        quit_events = []
        work_queue = Queue()
        for work_package in work_packages:
            work_queue.put(work_package)
        files_to_process = set(work_package[0] for work_package in work_packages)
        results_queue = Queue()

        for i in range(num_workers):
            quit_event = Event()
            pbar_pos = i+1
            p = Process(target=worker, args=(pbar_pos, work_queue, results_queue, quit_event))
            processes.append(p)
            quit_events.append(quit_event)
            p.start()
        with tqdm(total=len(files_to_process), desc="Files", position=0) as pbar:
            while files_to_process:
                try:
                    processed_file = results_queue.get(timeout=5)
                    files_to_process.remove(processed_file)
                    pbar.update()
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    for e in quit_events:
                        e.set()
                    for p in processes:
                        p.join()
        for p in processes:
            p.join()
        
    else:
        for ome_file in work_packages:
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
