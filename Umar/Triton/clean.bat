@echo off
:: Purpose: This script cleans the project directory by removing build artifacts, cache files, and temporary files
:: while preserving source files. It also reports statistics on what was cleaned.
:: Windows batch file version of clean.sh

:: Enable delayed expansion for variables inside loops
setlocal enabledelayedexpansion

:: Set colors for better readability (Windows console colors)
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "RED=[91m"
set "NC=[0m"

:: Initialize counters for statistics
set total_files_removed=0
set total_directories_removed=0
set cleaned_folders=

echo %YELLOW%Starting cleanup process...%NC%

:: Function to count files in a directory (implemented as a label)
goto :skip_count_files
:count_files
    set count=0
    for /r "%~1" %%F in (*) do set /a count+=1
    exit /b %count%
:skip_count_files

:: Section 1: Remove build directories
:: These are directories that contain compiled code and build artifacts
echo.
echo %YELLOW%Removing build directories...%NC%

:: Remove build directory
if exist "build" (
    call :count_files "build"
    set /a total_files_removed+=!errorlevel!
    set /a total_directories_removed+=1
    set "cleaned_folders=!cleaned_folders!build;"
    echo %GREEN%Removing directory:%NC% build (containing !errorlevel! files)
    rmdir /s /q "build"
)

:: Remove dist directory
if exist "dist" (
    call :count_files "dist"
    set /a total_files_removed+=!errorlevel!
    set /a total_directories_removed+=1
    set "cleaned_folders=!cleaned_folders!dist;"
    echo %GREEN%Removing directory:%NC% dist (containing !errorlevel! files)
    rmdir /s /q "dist"
)

:: Section 2: Remove Python cache files
:: Python creates cache files to speed up imports
echo.
echo %YELLOW%Removing Python cache files...%NC%

:: Count and remove __pycache__ directories
set pycache_count=0
set pycache_files=0

:: Find all __pycache__ directories
for /d /r "." %%d in (__pycache__) do (
    set /a pycache_count+=1
    call :count_files "%%d"
    set /a pycache_files+=!errorlevel!
    set "cleaned_folders=!cleaned_folders!%%d;"
)

:: Remove __pycache__ directories if any were found
if %pycache_count% gtr 0 (
    echo %GREEN%Removing %pycache_count% __pycache__ directories...%NC%
    set /a total_directories_removed+=%pycache_count%
    set /a total_files_removed+=%pycache_files%
    for /d /r "." %%d in (__pycache__) do rmdir /s /q "%%d" 2>nul
)

:: Function to remove files by pattern
goto :skip_remove_pattern
:remove_pattern
    set "pattern=%~1"
    set "description=%~2"
    set count=0
    
    :: Count matching files
    for /r "." %%f in (%pattern%) do (
        set /a count+=1
        set "parent_dir=%%~dpf"
        set "parent_dir=!parent_dir:~0,-1!"
        if "!cleaned_folders!" neq "!cleaned_folders:!parent_dir!;=!" (
            rem Directory already in list
        ) else (
            set "cleaned_folders=!cleaned_folders!!parent_dir!;"
        )
    )
    
    :: Remove files if any were found
    if !count! gtr 0 (
        echo %GREEN%Removing !count! %description%...%NC%
        set /a total_files_removed+=!count!
        for /r "." %%f in (%pattern%) do del "%%f" 2>nul
    )
    exit /b
:skip_remove_pattern

:: Remove various Python cache and build files
call :remove_pattern "*.pyc" "Python compiled files"          :: Compiled Python files
call :remove_pattern "*.pyo" "Python optimized files"         :: Optimized Python files
call :remove_pattern "*.pyd" "Python dynamic libraries"       :: Python dynamic libraries
:: Remove egg-info directories
for /d /r "." %%d in (*.egg-info) do (
    call :count_files "%%d"
    set /a total_files_removed+=!errorlevel!
    set /a total_directories_removed+=1
    set "cleaned_folders=!cleaned_folders!%%d;"
    echo %GREEN%Removing directory:%NC% %%d (containing !errorlevel! files)
    rmdir /s /q "%%d" 2>nul
)
call :remove_pattern "*.egg" "egg files"                      :: Python package distribution format

:: Section 3: Remove C/C++ build artifacts
:: These are files generated during C/C++ compilation
echo.
echo %YELLOW%Removing C/C++ build artifacts...%NC%
call :remove_pattern "*.o" "object files"                     :: Compiled object files
call :remove_pattern "*.a" "static libraries"                 :: Static libraries
call :remove_pattern "*.lo" "libtool object files"            :: Libtool object files
call :remove_pattern "*.la" "libtool archive files"           :: Libtool archive files
call :remove_pattern "*.so" "shared libraries"                :: Shared libraries (Linux/Unix)
call :remove_pattern "*.so.*" "shared libraries"              :: Shared libraries with version
call :remove_pattern "*.dylib" "dynamic libraries"            :: Dynamic libraries (macOS)
call :remove_pattern "*.dll" "dynamic link libraries"         :: Dynamic link libraries (Windows)
call :remove_pattern "*.exe" "executables"                    :: Executable files
call :remove_pattern "*.obj" "object files"                   :: Windows object files
call :remove_pattern "*.lib" "library files"                  :: Windows library files
call :remove_pattern "*.exp" "export files"                   :: Windows export files
call :remove_pattern "*.ilk" "incremental linking files"      :: Visual Studio incremental linking files
call :remove_pattern "*.pdb" "program database files"         :: Visual Studio debug info files

:: Section 4: Remove CMake artifacts
:: CMake generates these files during the build configuration process
echo.
echo %YELLOW%Removing CMake artifacts...%NC%
call :remove_pattern "CMakeCache.txt" "CMake cache files"     :: CMake cache file
:: Remove CMakeFiles directories
for /d /r "." %%d in (CMakeFiles) do (
    call :count_files "%%d"
    set /a total_files_removed+=!errorlevel!
    set /a total_directories_removed+=1
    set "cleaned_folders=!cleaned_folders!%%d;"
    echo %GREEN%Removing directory:%NC% %%d (containing !errorlevel! files)
    rmdir /s /q "%%d" 2>nul
)
call :remove_pattern "cmake_install.cmake" "CMake install files" :: CMake installation script
call :remove_pattern "Makefile" "Makefiles"                   :: Generated Makefiles

:: Section 5: Remove benchmark results
:: These files contain benchmark results that can be regenerated
:: Comment out this section if you want to keep benchmark results
echo.
echo %YELLOW%Removing benchmark results...%NC%
if exist "benchmark_results.txt" (
    echo %GREEN%Removing file:%NC% benchmark_results.txt
    del "benchmark_results.txt"
    set /a total_files_removed+=1
)
if exist "benchmark_results.png" (
    echo %GREEN%Removing file:%NC% benchmark_results.png
    del "benchmark_results.png"
    set /a total_files_removed+=1
)

:: Section 6: Remove temporary files
:: These are various temporary files created by editors and operating systems
echo.
echo %YELLOW%Removing temporary files...%NC%
call :remove_pattern "*.log" "log files"                      :: Log files
call :remove_pattern "*.tmp" "temporary files"                :: Temporary files
call :remove_pattern "*.swp" "vim swap files"                 :: Vim swap files
call :remove_pattern "*.swo" "vim swap files"                 :: Vim swap files
call :remove_pattern ".DS_Store" "macOS system files"         :: macOS directory metadata
call :remove_pattern "Thumbs.db" "Windows thumbnail cache"    :: Windows thumbnail cache

:: Print statistics about the cleanup
echo.
echo %BLUE%Cleanup Statistics:%NC%
echo %BLUE%Total files removed:%NC% %total_files_removed%
echo %BLUE%Total directories removed:%NC% %total_directories_removed%

:: Print list of cleaned folders
echo.
echo %BLUE%Cleaned folders:%NC%
for /f "tokens=1 delims=;" %%a in ("%cleaned_folders%") do (
    if not "%%a" == "" echo   - %%a
)

:: Print completion message and list preserved files
echo.
echo %GREEN%Cleanup complete!%NC%
echo %YELLOW%The following source files were preserved:%NC%

:: List all preserved source files
set "extensions=*.py *.cpp *.cu *.h *.hpp *.c *.md *.sh *.bat CMakeLists.txt"
for %%e in (%extensions%) do (
    for /r "." %%f in (%%e) do echo %%f
)

:: End of script
endlocal
