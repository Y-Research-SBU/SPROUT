import os
import cv2
import numpy as np
import scipy.io
import argparse
import random
from tqdm import tqdm

def inst_map_to_rgb(inst_map):
    # original dimensions
    height, width = inst_map.shape

    # RGB mask for visualization
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    used_colors = {}
    
    # Get unique instance IDs from the integer mask
    unique_ids = np.unique(inst_map)

    for idx in unique_ids:
        if idx == 0:
            continue
        
        if idx not in used_colors:
            color = tuple(random.randint(0, 255) for _ in range(3)) 
            used_colors[idx] = color
            
        rgb[inst_map == idx] = used_colors[idx]

    return rgb

def process_dataset(input_dir, output_dir, size, set_name):
    print(f"Processing '{set_name}' set...")

    # input paths
    img_input_path = os.path.join(input_dir, set_name, 'images')
    label_input_path = os.path.join(input_dir, set_name, 'labels')

    # output paths
    png_output_path = os.path.join(output_dir, set_name, 'png')
    instance_output_path = os.path.join(output_dir, set_name, 'instances')
    semantic_output_path = os.path.join(output_dir, set_name, 'semantics')

    os.makedirs(png_output_path, exist_ok=True)
    os.makedirs(instance_output_path, exist_ok=True)
    os.makedirs(semantic_output_path, exist_ok=True)

    image_ids = []
    
    # Get a sorted list of image files to ensure consistent order
    image_files = sorted([f for f in os.listdir(img_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for img_file in tqdm(image_files, desc=f"Processing {set_name} images"):
        base_name = os.path.splitext(img_file)[0]
        image_ids.append(img_file)

        # --- Process and save the image ---
        img_path = os.path.join(img_input_path, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
            
        resized_image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(png_output_path, f"{base_name}.png"), resized_image)

        # --- Process labels and create masks ---
        mat_path = os.path.join(label_input_path, f"{base_name}.mat")
        if not os.path.exists(mat_path):
            print(f"Warning: Label file not found for {img_file}. Skipping masks.")
            continue

        try:
            mat_data = scipy.io.loadmat(mat_path)
            # Load the instance map directly
            inst_map = mat_data.get('inst_map') 
            if inst_map is None:
                print(f"Warning: Could not find 'inst_map' key in {mat_path}. Skipping masks.")
                continue
        except Exception as e:
            print(f"Error loading {mat_path}: {e}. Skipping masks.")
            continue

        # --- Create masks at original size ---
        instance_mask_orig = inst_map_to_rgb(inst_map)
        
        gray_instance_mask_orig = cv2.cvtColor(instance_mask_orig, cv2.COLOR_BGR2GRAY)
        _, semantic_mask_orig = cv2.threshold(gray_instance_mask_orig, 1, 255, cv2.THRESH_BINARY)

        # --- Resize masks to target size ---
        instance_mask_resized = cv2.resize(instance_mask_orig, (size, size), interpolation=cv2.INTER_NEAREST)
        semantic_mask_resized = cv2.resize(semantic_mask_orig, (size, size), interpolation=cv2.INTER_NEAREST)

        # --- Save the resized masks ---
        cv2.imwrite(os.path.join(instance_output_path, f"{base_name}.png"), instance_mask_resized)
        cv2.imwrite(os.path.join(semantic_output_path, f"{base_name}.png"), semantic_mask_resized)


    # --- Save the list of image IDs ---
    txt_path = os.path.join(output_dir, f"{set_name}.txt")
    with open(txt_path, 'w') as f:
        for img_id in image_ids:
            f.write(f"{img_id}\n")
    print(f"Saved image IDs to {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Process an image dataset by resizing images and converting .mat labels to instance and semantic masks.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the root directory of the input dataset.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory where processed data will be saved.')
    parser.add_argument('--size', type=int, default=512, help='The target size (width and height) for the output images and masks.')
    
    args = parser.parse_args()

    # Process both the training and testing sets
    for set_name in ['train', 'test']:
        set_path = os.path.join(args.input_dir, set_name)
        if os.path.exists(set_path):
            process_dataset(args.input_dir, args.output_dir, args.size, set_name)
        else:
            print(f"Directory for '{set_name}' set not found at {set_path}. Skipping.")


if __name__ == '__main__':
    main()
