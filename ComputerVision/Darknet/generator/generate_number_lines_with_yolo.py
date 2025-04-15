"""
Number Line Dataset Generator with YOLO Annotations

This script generates a dataset of images with lines of random numbers (0-9)
and corresponding YOLO format annotations. Each image contains 3 lines of text,
with 10 numbers per line.

Usage:
    python generate_number_lines_with_yolo.py [options]

Options:
    --output_dir OUTPUT_DIR     Output directory
    --num_images NUM_IMAGES     Number of images to generate
    --image_width IMAGE_WIDTH   Image width
    --image_height IMAGE_HEIGHT Image height
    --font_dir FONT_DIR         Directory containing font files
"""

import os
import random
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    parser = argparse.ArgumentParser(description='Generate a dataset of images with lines of random numbers (0-9) and YOLO annotations')
    parser.add_argument('--output_dir', type=str, default='data/number_lines_yolo_fixed', help='Output directory')
    parser.add_argument('--num_images', type=int, default=20, help='Number of images to generate')
    parser.add_argument('--image_width', type=int, default=800, help='Image width')
    parser.add_argument('--image_height', type=int, default=600, help='Image height')
    parser.add_argument('--font_dir', type=str, default='C:/Windows/Fonts', help='Directory containing font files')
    return parser.parse_args()

def get_fonts(font_dir):
    """Get a list of font files from the specified directory"""
    import glob

    # Common font extensions
    font_extensions = ['*.ttf', '*.otf', '*.ttc']

    # Get all font files
    font_files = []
    for ext in font_extensions:
        font_files.extend(glob.glob(os.path.join(font_dir, ext)))

    if not font_files:
        print(f"Warning: No font files found in {font_dir}. Using default font.")
        return [None]  # Return a list with None to use default font

    print(f"Found {len(font_files)} font files in {font_dir}.")
    return font_files

def generate_background(width, height):
    """Generate a background image"""
    # Create a white background
    img = Image.new('RGB', (width, height), color='white')

    # Randomly add noise to the background
    if random.random() < 0.3:
        pixels = np.array(img)
        noise = np.random.randint(200, 256, (height, width, 3))
        pixels = noise.astype(np.uint8)
        img = Image.fromarray(pixels)

    return img

def generate_random_numbers_0to9(length=10):
    """Generate a string of random numbers from 0 to 9"""
    return ''.join(str(random.randint(0, 9)) for _ in range(length))

def generate_image_with_number_lines(img_index, font_files, args):
    """Generate an image with 3 lines of random numbers (0-9) and YOLO annotations"""
    # Create image filename
    img_filename = f"number_lines_{img_index:03d}.jpg"
    img_path = os.path.join(args.output_dir, img_filename)
    
    # Create annotation filename (same name as image but with .txt extension)
    ann_filename = f"number_lines_{img_index:03d}.txt"
    ann_path = os.path.join(args.output_dir, ann_filename)
    
    # Create a separate file to store the actual numbers (for reference)
    numbers_filename = f"numbers_{img_index:03d}.txt"
    numbers_path = os.path.join(args.output_dir, "numbers", numbers_filename)

    # Generate background
    img = generate_background(args.image_width, args.image_height)
    draw = ImageDraw.Draw(img)

    # Choose a random font
    font_path = random.choice(font_files)
    
    # Calculate font size based on image height and number of lines
    font_size = int(args.image_height / 10)  # Adjust as needed
    
    # Load font
    try:
        if font_path is None:
            font = ImageFont.load_default()
            print(f"Using default font with size {font_size}.")
        else:
            font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        # If the chosen font fails, try Arial as a fallback
        try:
            fallback_path = os.path.join(args.font_dir, 'arial.ttf')
            font = ImageFont.truetype(fallback_path, font_size)
        except:
            # If Arial fails too, use the default font
            font = ImageFont.load_default()
            print(f"Using default font with size {font_size}.")

    # Generate 3 lines of random numbers (0-9)
    lines = []
    annotations = []
    
    # Calculate line height
    line_height = int(args.image_height / 4)  # Divide height by 4 to get 3 lines with spacing
    
    for i in range(3):
        # Generate random numbers
        line = generate_random_numbers_0to9(10)
        lines.append(line)
        
        # Calculate position
        y = line_height * (i + 0.5)  # Center each line in its section
        
        # Calculate text width to center horizontally
        try:
            text_bbox = draw.textbbox((0, 0), line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except:
            # For older PIL versions
            try:
                text_width, text_height = draw.textsize(line, font=font)
            except:
                text_width = len(line) * font_size // 2  # Rough estimate
                text_height = font_size  # Rough estimate
        
        x = (args.image_width - text_width) // 2  # Center text horizontally
        
        # Draw text
        draw.text((x, y), line, fill='black', font=font)
        
        # Calculate YOLO format annotation
        # YOLO format: <class_id> <x_center> <y_center> <width> <height>
        # All values are normalized to [0, 1]
        class_id = 0  # Class ID for text line
        x_center = 0.5  # Center of the image horizontally
        y_center = y / args.image_height  # Normalized y-coordinate
        width = text_width / args.image_width  # Normalized width
        height = text_height / args.image_height  # Normalized height
        
        # Add annotation
        annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save image
    img.save(img_path)
    
    # Save YOLO annotations directly in the same directory as the image
    with open(ann_path, 'w') as f:
        f.write('\n'.join(annotations))
    
    # Save the actual numbers to a separate file (for reference)
    os.makedirs(os.path.dirname(numbers_path), exist_ok=True)
    with open(numbers_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return img_path, lines, annotations

def create_dataset_files(args):
    """Create dataset files for YOLO training"""
    # Create train.txt and valid.txt
    train_file = os.path.join(args.output_dir, "train.txt")
    valid_file = os.path.join(args.output_dir, "valid.txt")
    
    # Get all image paths
    image_paths = [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if f.endswith('.jpg')]
    
    # Shuffle and split
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * 0.8)  # 80% for training
    train_paths = image_paths[:split_idx]
    valid_paths = image_paths[split_idx:]
    
    # Write train.txt
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_paths))
    
    # Write valid.txt
    with open(valid_file, 'w') as f:
        f.write('\n'.join(valid_paths))
    
    # Create data file
    data_file = os.path.join(args.output_dir, "number_lines.data")
    with open(data_file, 'w') as f:
        f.write(f"classes = 1\n")  # 1 class: text line
        f.write(f"train = {os.path.abspath(train_file)}\n")
        f.write(f"valid = {os.path.abspath(valid_file)}\n")
        f.write(f"names = {os.path.abspath(os.path.join(args.output_dir, 'classes.names'))}\n")
        f.write(f"backup = backup\n")
    
    # Create classes.names file
    names_file = os.path.join(args.output_dir, "classes.names")
    with open(names_file, 'w') as f:
        f.write("text_line\n")  # Class name for text line

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "numbers"), exist_ok=True)
    
    # Get fonts
    font_files = get_fonts(args.font_dir)
    
    # Print configuration
    print(f"Number Line Dataset Generator with YOLO Annotations")
    print(f"=================================================")
    print(f"Using {len(font_files)} fonts")
    print(f"Generating {args.num_images} images with 3 lines of random numbers (0-9)...")
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.image_width}x{args.image_height}")
    
    # Generate images
    all_lines = []
    
    for i in range(args.num_images):
        img_path, lines, annotations = generate_image_with_number_lines(i, font_files, args)
        all_lines.append((img_path, lines, annotations))
        
        print(f"Generated image {i+1}/{args.num_images}: {img_path}")
        print(f"  Numbers:")
        for j, line in enumerate(lines):
            print(f"    Line {j+1}: {line}")
        print(f"  YOLO Annotations:")
        for ann in annotations:
            print(f"    {ann}")
    
    # Create dataset files
    create_dataset_files(args)
    
    print(f"\nDataset generation complete!")
    print(f"Generated {args.num_images} images with 3 lines of random numbers (0-9)")
    print(f"Files saved to {args.output_dir}")
    print(f"YOLO annotations saved directly with each image")
    print(f"Original numbers saved to {os.path.join(args.output_dir, 'numbers')}")
    print(f"Dataset files created: train.txt, valid.txt, number_lines.data, classes.names")

if __name__ == "__main__":
    main()
