#!/bin/bash

# Purpose: This script cleans the project directory by removing build artifacts, cache files, and temporary files
# while preserving source files. It also reports statistics on what was cleaned.

# ANSI color codes for better readability
GREEN='\033[0;32m'   # Used for success messages and file counts
YELLOW='\033[1;33m'  # Used for section headers
BLUE='\033[0;34m'    # Used for statistics
RED='\033[0;31m'     # Used for warnings or errors
NC='\033[0m'         # No Color - resets text formatting

# Initialize counters for statistics
total_files_removed=0
total_directories_removed=0
cleaned_folders=()

echo -e "${YELLOW}Starting cleanup process...${NC}"

# Function to safely remove files/directories and update counters
safe_remove() {
    # $1: path to file or directory to remove
    if [ -e "$1" ]; then
        if [ -d "$1" ]; then
            # Count files in directory before removing
            local files_count=$(find "$1" -type f | wc -l)
            total_files_removed=$((total_files_removed + files_count))
            total_directories_removed=$((total_directories_removed + 1))
            cleaned_folders+=("$1")
            echo -e "${GREEN}Removing directory:${NC} $1 (containing $files_count files)"
        else
            total_files_removed=$((total_files_removed + 1))
            echo -e "${GREEN}Removing file:${NC} $1"
        fi
        rm -rf "$1"
    fi
}

# Function to count and remove files matching a pattern
remove_pattern() {
    # $1: file pattern to match (e.g., "*.pyc")
    # $2: description of files for output
    local pattern=$1
    local description=$2

    # Find and count matching files
    local files=($(find . -name "$pattern" 2>/dev/null))
    local count=${#files[@]}

    if [ "$count" -gt 0 ]; then
        echo -e "${GREEN}Removing ${count} ${description}...${NC}"

        # Track unique parent directories for statistics
        local dirs=()
        for file in "${files[@]}"; do
            local dir=$(dirname "$file")
            if [[ ! " ${cleaned_folders[*]} " =~ " $dir " ]]; then
                cleaned_folders+=("$dir")
            fi
        done

        # Remove the files
        find . -name "$pattern" -exec rm -rf {} \; 2>/dev/null || true
        total_files_removed=$((total_files_removed + count))
    fi
}

# Section 1: Remove build directories
# These are directories that contain compiled code and build artifacts
echo -e "\n${YELLOW}Removing build directories...${NC}"
safe_remove "./build"    # Main build directory for C++ code
safe_remove "./dist"     # Distribution directory for Python packages

# Section 2: Remove Python cache files
# Python creates cache files to speed up imports
echo -e "\n${YELLOW}Removing Python cache files...${NC}"

# Remove __pycache__ directories which contain compiled Python files
pycache_dirs=$(find . -type d -name "__pycache__" 2>/dev/null)
pycache_count=$(echo "$pycache_dirs" | grep -v '^$' | wc -l)
if [ "$pycache_count" -gt 0 ]; then
    echo -e "${GREEN}Removing $pycache_count __pycache__ directories...${NC}"
    total_directories_removed=$((total_directories_removed + pycache_count))
    # Count files in pycache directories
    for dir in $pycache_dirs; do
        if [ -d "$dir" ]; then
            files_count=$(find "$dir" -type f | wc -l)
            total_files_removed=$((total_files_removed + files_count))
            cleaned_folders+=("$dir")
        fi
    done
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
fi

# Remove various Python cache and build files
remove_pattern "*.pyc" "Python compiled files"          # Compiled Python files
remove_pattern "*.pyo" "Python optimized files"         # Optimized Python files
remove_pattern "*.pyd" "Python dynamic libraries"       # Python dynamic libraries
remove_pattern "*.so" "shared object files"             # Shared object files (compiled extensions)
remove_pattern "*.egg-info" "egg info directories"      # Python package metadata
remove_pattern "*.egg" "egg files"                      # Python package distribution format

# Section 3: Remove C/C++ build artifacts
# These are files generated during C/C++ compilation
echo -e "\n${YELLOW}Removing C/C++ build artifacts...${NC}"
remove_pattern "*.o" "object files"                     # Compiled object files
remove_pattern "*.a" "static libraries"                 # Static libraries
remove_pattern "*.lo" "libtool object files"            # Libtool object files
remove_pattern "*.la" "libtool archive files"           # Libtool archive files
remove_pattern "*.so*" "shared libraries"               # Shared libraries (Linux/Unix)
remove_pattern "*.dylib" "dynamic libraries"            # Dynamic libraries (macOS)
remove_pattern "*.dll" "dynamic link libraries"         # Dynamic link libraries (Windows)
remove_pattern "*.exe" "executables"                    # Executable files

# Section 4: Remove CMake artifacts
# CMake generates these files during the build configuration process
echo -e "\n${YELLOW}Removing CMake artifacts...${NC}"
remove_pattern "CMakeCache.txt" "CMake cache files"     # CMake cache file
remove_pattern "CMakeFiles" "CMake files directories"   # Directory with CMake temporary files
remove_pattern "cmake_install.cmake" "CMake install files" # CMake installation script
remove_pattern "Makefile" "Makefiles"                   # Generated Makefiles

# Section 5: Remove benchmark results
# These files contain benchmark results that can be regenerated
# Comment out this section if you want to keep benchmark results
echo -e "\n${YELLOW}Removing benchmark results...${NC}"
safe_remove "./benchmark_results.txt"                   # Text file with benchmark results
safe_remove "./benchmark_results.png"                   # Visualization of benchmark results

# Section 6: Remove temporary files
# These are various temporary files created by editors and operating systems
echo -e "\n${YELLOW}Removing temporary files...${NC}"
remove_pattern "*.log" "log files"                      # Log files
remove_pattern "*.tmp" "temporary files"                # Temporary files
remove_pattern "*.swp" "vim swap files"                 # Vim swap files
remove_pattern "*.swo" "vim swap files"                 # Vim swap files
remove_pattern ".DS_Store" "macOS system files"         # macOS directory metadata
remove_pattern "Thumbs.db" "Windows system files"       # Windows thumbnail cache

# Print statistics about the cleanup
echo -e "\n${BLUE}Cleanup Statistics:${NC}"
echo -e "${BLUE}Total files removed:${NC} $total_files_removed"
echo -e "${BLUE}Total directories removed:${NC} $total_directories_removed"

# Print list of cleaned folders
if [ ${#cleaned_folders[@]} -gt 0 ]; then
    echo -e "\n${BLUE}Cleaned folders:${NC}"
    for folder in "${cleaned_folders[@]}"; do
        echo "  - $folder"
    done
fi

# Print completion message and list preserved files
echo -e "\n${GREEN}Cleanup complete!${NC}"
echo -e "${YELLOW}The following source files were preserved:${NC}"
find . -type f \( -name "*.py" -o -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" -o -name "*.c" -o -name "*.md" -o -name "*.sh" -o -name "CMakeLists.txt" \) | sort
