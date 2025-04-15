# Dataset Generator Scripts

This folder contains scripts for generating datasets of images with lines of random numbers and corresponding YOLO annotations.

## Scripts

### 1. generate_number_lines.py
- Generates images with 3 lines of random numbers (0-1)
- Each line contains 10 binary digits
- Saves the actual numbers in text files with the same name as the images

### 2. generate_number_lines_0to9.py
- Generates images with 3 lines of random numbers (0-9)
- Each line contains 10 digits
- Saves the actual numbers in text files with the same name as the images

### 3. generate_number_lines_with_yolo.py
- Generates images with 3 lines of random numbers (0-9)
- Each line contains 10 digits
- Saves YOLO annotations for each line of text (line-level annotations)
- Class ID is 0 for all lines (single class: "text_line")
- Saves the actual numbers in a separate "numbers" directory for reference

### 4. generate_number_lines_with_character_annotations.py
- Generates images with 3 lines of random numbers (0-9)
- Each line contains 10 digits
- Saves YOLO annotations for each individual character (character-level annotations)
- Class IDs are the digits themselves (0-9)
- Saves the actual numbers in a separate "numbers" directory for reference

## Usage

All scripts accept the following command-line arguments:

```
--output_dir OUTPUT_DIR     Output directory
--num_images NUM_IMAGES     Number of images to generate
--image_width IMAGE_WIDTH   Image width
--image_height IMAGE_HEIGHT Image height
--font_dir FONT_DIR         Directory containing font files
```

Example usage:

```bash
python generator/generate_number_lines_with_character_annotations.py --num_images 20 --output_dir data/my_dataset
```

## Output

Each script creates a dataset with the following structure:

- Images: `.jpg` files
- Annotations: `.txt` files (YOLO format)
- Dataset configuration files:
  - `classes.names`: Class names
  - `number_lines.data`: Dataset configuration
  - `train.txt`: List of training images
  - `valid.txt`: List of validation images

The character-level annotations script (`generate_number_lines_with_character_annotations.py`) is particularly useful for training YOLO models to detect and recognize individual digits, which is a key component of the OCR pipeline.
