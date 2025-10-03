#!/usr/bin/env python3
import os
import argparse
import shutil
import cv2
import numpy as np
import random

def main(args):
    input_dir = args.input_dir
    png_dir = os.path.join(input_dir, "png")
    semantics_dir = os.path.join(input_dir, "semantics")
    instances_dir = os.path.join(input_dir, "instances")

    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(semantics_dir, exist_ok=True)
    os.makedirs(instances_dir, exist_ok=True)

    # Move all images from Slide_* folders to png/
    for entry in os.listdir(input_dir):
        path = os.path.join(input_dir, entry)
        if os.path.isdir(path) and entry.lower().startswith("slide"):
            for fname in os.listdir(path):
                if fname.lower().endswith(".png"):
                    shutil.move(os.path.join(path, fname), os.path.join(png_dir, fname))

    # Move all semantic masks from GT_* folders to semantics/
    for entry in os.listdir(input_dir):
        path = os.path.join(input_dir, entry)
        if os.path.isdir(path) and entry.lower().startswith("gt"):
            for fname in os.listdir(path):
                if fname.lower().endswith(".png"):
                    shutil.move(os.path.join(path, fname), os.path.join(semantics_dir, fname))

    # Generate colorful instance masks
    for fname in os.listdir(semantics_dir):
        mask_path = os.path.join(semantics_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        h, w = mask.shape
        inst = np.zeros((h, w, 3), dtype=np.uint8)

        # Find connected components to separate instances
        num_labels, labels = cv2.connectedComponents(mask)
        if num_labels <= 1:
            cv2.imwrite(os.path.join(instances_dir, fname), inst)
            continue

        # Assign a unique random color to each instance label
        instance_ids = list(range(1, num_labels))
        color_map = {}
        used_colors = set()
        for uid in instance_ids:
            while True:
                color = (random.randint(0, 255), 
                         random.randint(0, 255), 
                         random.randint(0, 255))
                if color not in used_colors:
                    used_colors.add(color)
                    color_map[uid] = color
                    break

        # Paint each connected component with its unique random color
        for uid, color in color_map.items():
            inst[labels == uid] = color

        out_path = os.path.join(instances_dir, fname)
        cv2.imwrite(out_path, inst)

    # Delete empty GT_* and Slide_* folders
    for entry in os.listdir(input_dir):
        path = os.path.join(input_dir, entry)
        if os.path.isdir(path) and (entry.lower().startswith("gt_") or entry.lower().startswith("slide_")):
            if not os.listdir(path):
                os.rmdir(path)

    # Write index file
    index_path = os.path.join(input_dir, f"{args.fname}.txt")
    with open(index_path, "w") as f:
        for fname in sorted(os.listdir(png_dir)):
            if fname.lower().endswith(".png"):
                f.write(f"{fname}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize dataset: move images and semantic masks, generate colorful instance masks, and index filenames."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Path to the root of the dataset containing GT_xx and Slide_xx folders."
    )
    parser.add_argument(
        "--fname", required=True,
        help="Name of the output index txt file."
    )
    args = parser.parse_args()

    main(args)