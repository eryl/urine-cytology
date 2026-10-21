import argparse
from pathlib import Path
import multiprocessing
import re

from tqdm import tqdm
from imageio.v2 import imread, imwrite


def boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    return not (box1['max_x'] < box2['min_x'] or
                box1['min_x'] > box2['max_x'] or
                box1['max_y'] < box2['min_y'] or
                box1['min_y'] > box2['max_y'])

def merge_two_boxes(box1, box2):
    """Merge two overlapping bounding boxes."""
    min_x = min(box1['min_x'], box2['min_x'])
    max_x = max(box1['max_x'], box2['max_x'])
    min_y = min(box1['min_y'], box2['min_y'])
    max_y = max(box1['max_y'], box2['max_y'])
    return {
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'width': max_x - min_x,
        'height': max_y - min_y
    }


def merge_overlapping_bboxes(bboxes):
    merged = []

    while bboxes:
        box = bboxes.pop()
        did_merge = False
        unmerged_boxes = []
        
        for other in bboxes:
            if boxes_overlap(box, other):
                box = merge_two_boxes(box, other)
                did_merge = True
            else:
                unmerged_boxes.append(other)

        if did_merge:
            # There may be other boxes that now overlap with the new merged one
            bboxes = unmerged_boxes + [box]
        
        else:
            merged.append(box)

    return merged


def process_work_package(work_package):
    image_path = work_package["image_path"]
    segmentation_path = work_package["segmentation_path"]
    output_dir = work_package["output_dir"]
    margin = work_package["margin"]
    min_area = work_package["min_area"]
    image = imread(image_path)
    image_height, image_width = image.shape[:2]
    
    with open(segmentation_path) as fp:
        bboxes = []
        for line in fp:
            try:
                int_points = [int(c) for c in line.strip().split(',')]
            except ValueError:
                # Some images have no regions, 
                # so there will be no non-empty 
                # lines above, which throws a 
                # ValueError when passed to int()
                return
            x = int_points[::2]
            y = int_points[1::2]
            max_x = max(x)
            min_x = min(x)
            max_y = max(y)
            min_y = min(y)
            width = max_x - min_x
            height = max_y - min_y
            area = width*height
            
            if area >= min_area:
                max_x = min(max_x + margin, image_width)
                min_x = max(min_x - margin, 0)
                max_y = min(max_y + margin, image_height)
                min_y = max(min_y - margin, 0)
                
                bbox = {'max_x': max_x, 
                        'min_x': min_x, 
                        'max_y': max_y, 
                        'min_y': min_y, 
                        'width': width, 
                        'height': height} 
                bboxes.append(bbox)
    #print(f"Number of bboxes before merge: {len(bboxes)}")
    merged_bboxes = merge_overlapping_bboxes(bboxes)
    #print(f"Number of bboxes after merge: {len(merged_bboxes)}")
    image_name = image_path.with_suffix('').name
    for bbox in merged_bboxes:
        min_y, max_y, min_x, max_x = bbox['min_y'], bbox['max_y'], bbox['min_x'], bbox['max_x']
        region = image[min_y:max_y, min_x:max_x]
        output_filename = f"{image_name}_y{min_y}-{max_y}_x{min_x}-{max_x}.png"
        output_path = output_dir / output_filename
        imwrite(output_path, region)
        

def main():
    parser = argparse.ArgumentParser(description="Extract region from ImageJ plygon text files produced by cellpose")
    parser.add_argument("image_directory", help="Directory containing the images and region files", type=Path)
    parser.add_argument("output_directory", help="Directory to save the regions to", type=Path)
    parser.add_argument("--num-workers", help="Number of parallell processes to use", type=int, default=1)
    parser.add_argument("--margin", help="How many pixels will be added to each side of each bounding box", type=int, default=50)
    parser.add_argument("--min-area", help="bounding boxes with area smaller than this will be discarded", type=int, default=400)
    args = parser.parse_args()
    
    output_dir = args.output_directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    segmentations_paths = sorted(args.image_directory.glob("**/*.txt"))
    work_packages = []
    for segmentation_path in segmentations_paths:
        m = re.match(r'^(.+?)_cp_outlines\.txt$', segmentation_path.name)
        if m is None:
            print(f"Couldnt match name {segmentation_path.name}")
            continue
        
        base_name, = m.groups()
        image_path = segmentation_path.with_name(f"{base_name}.png")
        if image_path.exists():
            work_packages.append({'image_path': image_path, 
                                  'segmentation_path': segmentation_path, 
                                  "margin": args.margin, 
                                  "min_area": args.min_area,
                                  "output_dir": output_dir})
    
    if args.num_workers > 1:
        with multiprocessing.Pool(args.num_workers) as pool:
            for result in tqdm(pool.imap_unordered(process_work_package, work_packages), 
                               total=len(work_packages), 
                               desc="Work packages"):
                pass
    else:
        for result in tqdm(map(process_work_package, work_packages), 
                               total=len(work_packages), 
                               desc="Work packages"):
                pass
                
    

if __name__ == '__main__':
    main()