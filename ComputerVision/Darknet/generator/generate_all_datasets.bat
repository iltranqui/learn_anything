@echo off
REM =========================================================================
REM generate_all_datasets.bat
REM =========================================================================
REM This batch file runs all the dataset generation scripts to create
REM different types of datasets for OCR training and testing.
REM
REM It will generate:
REM 1. Binary lines dataset (0-1 digits)
REM 2. Number lines dataset (0-9 digits)
REM 3. Line-level annotations dataset (YOLO annotations for text lines)
REM 4. Character-level annotations dataset (YOLO annotations for individual digits)
REM
REM Each dataset will be saved in its own directory.
REM =========================================================================

echo =========================================================================
echo Generating all datasets for OCR training and testing
echo =========================================================================
echo.

REM Generate binary lines dataset
echo Step 1/4: Generating binary lines dataset...
call generate_binary_lines.bat
echo.

REM Generate number lines dataset
echo Step 2/4: Generating number lines dataset...
call generate_number_lines.bat
echo.

REM Generate line-level annotations dataset
echo Step 3/4: Generating line-level annotations dataset...
call generate_line_annotations.bat
echo.

REM Generate character-level annotations dataset
echo Step 4/4: Generating character-level annotations dataset...
call generate_character_annotations.bat
echo.

echo =========================================================================
echo All datasets generated successfully!
echo =========================================================================
echo.
echo Datasets:
echo - Binary lines: ..\data\binary_lines
echo - Number lines: ..\data\number_lines_0to9
echo - Line annotations: ..\data\line_annotations
echo - Character annotations: ..\data\character_annotations
echo.
