"""
Number Line Dataset Generator (0-9)

This script generates a dataset of images with lines of random numbers (0-9).
Each image contains 3 lines of text, with 10 numbers per line.

Usage:
    python generate_number_lines_0to9.py [options]

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
    parser = argparse.ArgumentParser(description='Generate a dataset of images with lines of random numbers (0-9)')
    parser.add_argument('--output_dir', type=str, default='data/number_lines_0to9', help='Output directory')
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
    """Generate an image with 3 lines of random numbers (0-9)"""
    # Create image filename
    img_filename = f"number_lines_{img_index:03d}.jpg"
    img_path = os.path.join(args.output_dir, img_filename)
    
    # Create text filename to save the generated numbers
    txt_filename = f"number_lines_{img_index:03d}.txt"
    txt_path = os.path.join(args.output_dir, txt_filename)

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
    for i in range(3):
        line = generate_random_numbers_0to9(10)
        lines.append(line)

    # Calculate line height
    line_height = int(args.image_height / 4)  # Divide height by 4 to get 3 lines with spacing
    
    # Draw lines of text
    for i, line in enumerate(lines):
        # Calculate position
        y = line_height * (i + 0.5)  # Center each line in its section
        
        # Calculate text width to center horizontally
        try:
            text_bbox = draw.textbbox((0, 0), line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
        except:
            # For older PIL versions
            try:
                text_width, _ = draw.textsize(line, font=font)
            except:
                text_width = len(line) * font_size // 2  # Rough estimate
        
        x = (args.image_width - text_width) // 2  # Center text horizontally
        
        # Draw text
        draw.text((x, y), line, fill='black', font=font)

    # Save image
    img.save(img_path)
    
    # Save the generated numbers to a text file
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return img_path, lines

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get fonts
    font_files = get_fonts(args.font_dir)
    
    # Print configuration
    print(f"Number Line Dataset Generator (0-9)")
    print(f"==================================")
    print(f"Using {len(font_files)} fonts")
    print(f"Generating {args.num_images} images with 3 lines of random numbers (0-9)...")
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.image_width}x{args.image_height}")
    
    # Generate images
    all_lines = []
    
    for i in range(args.num_images):
        img_path, lines = generate_image_with_number_lines(i, font_files, args)
        all_lines.append((img_path, lines))
        
        print(f"Generated image {i+1}/{args.num_images}: {img_path}")
        for j, line in enumerate(lines):
            print(f"  Line {j+1}: {line}")
    
    print(f"\nDataset generation complete!")
    print(f"Generated {args.num_images} images with 3 lines of random numbers (0-9)")
    print(f"Files saved to {args.output_dir}")

if __name__ == "__main__":
    main()
