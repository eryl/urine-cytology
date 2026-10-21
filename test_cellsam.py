import argparse
from pathlib import Path

from torchvision.io import read_image
from cellSAM import segment_cellular_image, get_model

def main():
    parser = argparse.ArgumentParser(description="Test cell segmentation with cellsam")
    parser.add_argument("image_path", type=Path)

    args = parser.parse_args()

    model = get_model('cellsam_extra')
    model.to('cuda')
    image_path = args.image_path
    image = read_image(image_path).to('cuda')
    mask, _, _ = segment_cellular_image(image, model, device='cuda')
    pass

if __name__ == '__main__':
    main()