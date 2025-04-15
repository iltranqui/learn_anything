@echo off
REM =========================================================================
REM generate_binary_lines.bat
REM =========================================================================
REM This batch file runs the generate_number_lines.py script to create
REM a dataset of images with 3 lines of random binary digits (0-1).
REM Each image contains 3 lines with 10 binary digits per line.
REM
REM Output:
REM - Images (.jpg) with 3 lines of binary digits
REM - Text files (.txt) containing the actual digits
REM
REM Parameters:
REM - num_images: Number of images to generate (default: 20)
REM - output_dir: Directory to save the generated files (default: data/number_lines)
REM - image_width: Width of the generated images (default: 800)
REM - image_height: Height of the generated images (default: 600)
REM =========================================================================

REM Set Python path (modify if needed)
set PYTHON=python

REM Set script path
set SCRIPT=generate_number_lines.py

REM Set parameters
set NUM_IMAGES=20
set OUTPUT_DIR=..\data\binary_lines
set IMAGE_WIDTH=800
set IMAGE_HEIGHT=600

REM Create output directory if it doesn't exist
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Run the script
echo Generating %NUM_IMAGES% images with binary digits (0-1)...
%PYTHON% %SCRIPT% --num_images %NUM_IMAGES% --output_dir %OUTPUT_DIR% --image_width %IMAGE_WIDTH% --image_height %IMAGE_HEIGHT%

echo.
echo Done! Generated files are in %OUTPUT_DIR%
echo.
