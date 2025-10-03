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

def process_split(split_root: Path, out_root: Path, size: int) -> list[str]:
    """
    Process a split directory (train/ or test/).
    Returns the list of generated PNG filenames (basename only).
    """
    img_dir = split_root / "images"
    ann_dir = split_root / "annotations"

    inst_dir = out_root / "instances"
    sem_dir  = out_root / "semantics"
    png_dir  = out_root / "png"
    inst_dir.mkdir(parents=True, exist_ok=True)
    sem_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    png_fnames = []

    for img_path in sorted(img_dir.glob("*.tif")):
        stem = img_path.stem

        # 1) Resize and save PNG image
        rgb_resized = resize_tif_to_rgb(img_path, size)
        cv2.imwrite(
            str(png_dir / f"{stem}.png"),
            cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)
        )
        png_fnames.append(f"{stem}.png")

        # 2) Build masks if annotation exists
        xml_path = ann_dir / f"{stem}.xml"
        if not xml_path.exists():
            print(f"[!] No annotation for {stem} in {ann_dir}, skipping masks.")
            continue

        # Generate instance & semantic masks at original resolution first
        orig = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        h, w = orig.shape[:2]
        polys = read_annotation(xml_path)
        inst_mask_full = to_instance_mask(polys, h, w)
        sem_mask_full  = to_semantic_mask(inst_mask_full)

        # Resize masks to match resized image size (W,H order for cv2.resize)
        H, W, _ = rgb_resized.shape
        inst_mask = cv2.resize(inst_mask_full, (W, H), interpolation=cv2.INTER_NEAREST)
        sem_mask  = cv2.resize(sem_mask_full,  (W, H), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(inst_dir / f"{stem}.png"), inst_mask)
        cv2.imwrite(str(sem_dir / f"{stem}.png"), sem_mask)

    return png_fnames


def main():
    args = parse_args()
    root = args.input_dir
    out = args.output_dir or root

    # Expecting: root/{train,test}/{images,annotations}
    splits = ["train", "test"]
    for split in splits:
        split_root = root / split
        if not (split_root / "images").exists():
            print(f"[!] Missing '{split}/images' under {root}, skipping split.")
            continue
        if not (split_root / "annotations").exists():
            print(f"[!] Missing '{split}/annotations' under {root}, skipping split.")
            continue

        out_root = out / split
        print(f"Processing split: {split_root} → {out_root}")
        png_list = process_split(split_root, out_root, args.size)

        # Write split-specific index file (e.g., root/train.txt)
        index_path = root / f"{split}.txt"
        with open(index_path, "w") as f:
            f.write("\n".join(png_list))
        print(f"Wrote index: {index_path} ({len(png_list)} entries)")


if __name__ == "__main__":
    main()
