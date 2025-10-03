#!/usr/bin/env python3
import math
import random
import argparse
import cv2
import numpy as np
from pathlib import Path
from lxml import etree

random.seed(42)


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert MoNuSeg XML → instance & semantic masks, "
                    "and resize TIFF images to PNG."
    )
    p.add_argument(
        "-i", "--input_dir", type=Path, required=True,
        help="MoNuSeg root (must contain 'images/' & 'annotations/')"
    )
    p.add_argument(
        "-o", "--output_dir", type=Path, default=None,
        help="Where to write 'instance', 'semantic', and 'png' subfolders "
             "(default: same as input_dir)"
    )
    p.add_argument(
        "-s", "--size", type=int, default=1024,
        help="Resize dimensions for PNG images"
    )
    p.add_argument(
        "-f", "--fname", type=str, required=True,
        help="txt index file name"
    )
    return p.parse_args()

def resize_tif_to_rgb(path: Path, size: int) -> np.ndarray:
    """Load a TIFF, resize, and return an RGB array."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_annotation(path: Path) -> list[np.ndarray]:
    """Parse MoNuSeg XML → list of N×2 int32 polygons."""
    tree   = etree.parse(str(path))
    regions = tree.xpath("/Annotations/Annotation/Regions/Region")
    annotations = []
    for region in regions:
        pts = []
        for v in region.xpath("Vertices/Vertex"):
            x = math.floor(float(v.attrib["X"]))
            y = math.floor(float(v.attrib["Y"]))
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        if pts.shape[0] >= 3:
            annotations.append(pts)
    return annotations

def to_instance_mask(annotations, height, width) -> np.ndarray:
    """
    Build RGB mask: each polygon filled with a random color.
    """
    inst = np.zeros((height, width), dtype=np.uint16)
    for i, poly in enumerate(annotations, start=1):
        cv2.drawContours(inst, [poly], contourIdx=-1,
                         color=i, thickness=cv2.FILLED)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    used_colors = {}
    for idx in np.unique(inst):
        if idx == 0:
            continue
        if idx not in used_colors:
            color = tuple(random.randint(0, 255) for _ in range(3))
            used_colors[idx] = color
        rgb[inst == idx] = used_colors[idx]

    return rgb


def to_semantic_mask(instance_mask) -> np.ndarray:
    """Convert instance-RGB → binary mask (0 or 255)."""
    fg = np.any(instance_mask != 0, axis=2)
    return (fg.astype(np.uint8) * 255)

def main():
    args = parse_args()
    root = args.input_dir
    out = args.output_dir or root
    txt_name = args.fname

    img_dir = root / "images"
    ann_dir = root / "annotations"
    inst_dir = (out / "instances");   inst_dir.mkdir(parents=True, exist_ok=True)
    sem_dir  = (out / "semantics");    sem_dir.mkdir(parents=True, exist_ok=True)
    png_dir  = (out / "png");         png_dir.mkdir(parents=True, exist_ok=True)

    png_fnames = []

    for img_path in sorted(img_dir.glob("*.tif")):
        stem = img_path.stem
        rgb = resize_tif_to_rgb(img_path, args.size)

        cv2.imwrite(str(png_dir / f"{stem}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        png_name = f"{stem}.png"
        png_fnames.append(png_name)

        # Load annotations
        xml_path = ann_dir / f"{stem}.xml"
        if not xml_path.exists():
            print(f"[!] No annotation for {stem}, skipping masks.")
            continue

        orig = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        h, w = orig.shape[:2]  
        polys = read_annotation(xml_path)
        inst_mask = to_instance_mask(polys, h, w)
        sem_mask  = to_semantic_mask(inst_mask)

        H, W, _ = rgb.shape
        inst_mask = cv2.resize(inst_mask, (H, W), interpolation=cv2.INTER_NEAREST)
        sem_mask  = cv2.resize(sem_mask,  (H, W), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(inst_dir / f"{stem}.png"), inst_mask)
        cv2.imwrite(str(sem_dir  / f"{stem}.png"), sem_mask)

    index_path = Path(args.input_dir).parent / f"{txt_name}.txt"
    with open(index_path, 'w') as f:
        f.write('\n'.join(png_fnames))



if __name__ == "__main__":
    main()
