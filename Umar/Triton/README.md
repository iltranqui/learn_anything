# Conv3D + GroupNorm + Mean Benchmark

This project compares the performance of a 3D convolution followed by group normalization and mean computation, implemented in:
1. PyTorch (Python)
2. CUDA (C++/CUDA)
3. LibTorch (C++)

## Files

- `23_Conv3d_GroupNorm_Mean.py`: PyTorch implementation
- `23_Conv3d_GroupNorm_Mean.cu`: CUDA implementation
- `benchmark.py`: Comprehensive benchmark script for all implementations
- `libtorch_algorithm.cpp`: C++ implementation with custom algorithm
- `CMakeLists.txt`: CMake configuration for LibTorch implementation
- `run_benchmark.sh`: Shell script to run the benchmark

## Setup and Installation

### Prerequisites

- Python 3.11 or higher
- PyTorch
- CUDA toolkit
- LibTorch (C++ distribution of PyTorch)
- CMake
- C++ compiler with CUDA support

### Python Environment Setup

The project uses a Python virtual environment. To activate it:

```bash
source .venv/bin/activate

# Install required packages
pip install torch torchvision setuptools matplotlib
```

### Building the CUDA Extension

```bash
cd /home/kerrigan/learn_anything/Umar/Triton
python setup.py install
```

### Building the LibTorch Implementation

```bash
# Download and extract LibTorch (if not already done)
mkdir -p ~/libtorch
cd ~/libtorch
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip

# Build the LibTorch implementation
cd /home/kerrigan/learn_anything/Umar/Triton
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/home/kerrigan/libtorch/libtorch"
make
```

## Running the Benchmarks

To run all benchmarks, simply execute the provided shell script:

```bash
./run_benchmark.sh
```

This will:
1. Build the C++ implementation
2. Run the Python benchmark script which benchmarks all implementations
3. Generate results and visualizations

Alternatively, you can run the Python benchmark directly:

```bash
python benchmark.py
```

## Results

We've successfully benchmarked different implementations of the 3D convolution + GroupNorm + Mean operation using PyTorch 2.6.0 with CUDA 12.4. Here's a summary of our findings:

| Implementation | Time (ms) | Relative Performance |
|----------------|-----------|---------------------|
| Custom Algorithm (C++) | 0.04 ms | Fastest (1.00x) |
| LibTorch (C++) | 6.26 ms | 156.50x slower |
| PyTorch Functional | 7.25 ms | 181.25x slower |
| PyTorch Module | 8.56 ms | 214.00x slower |
| Triton | 19.56 ms | 489.00x slower |

Note: The Custom Algorithm is a simplified implementation that only computes the mean of the group norm bias, not the full operation.

For detailed analysis, see the [FINAL_RESULTS.md](FINAL_RESULTS.md) file.

A visualization of the results is available in `benchmark_results.png`.

## Notes

- The CUDA implementation in this example is a simplified version that focuses on the group normalization bias and mean computation, not the full convolution operation.
- The LibTorch implementation includes both a PyTorch-like implementation and a CUDA kernel implementation for comparison.
- Performance may vary depending on hardware, CUDA version, and PyTorch version.

## Troubleshooting

If you encounter issues with PyTorch installation in the virtual environment:
- Try using `pip install --target=.venv/lib/python3.x/site-packages torch`
- Ensure you have CUDA installed if you want to use GPU acceleration

If you encounter issues with the CUDA extension:
- Ensure you have the CUDA toolkit installed
- Check that PyTorch was built with CUDA support

If you encounter issues with the LibTorch build:
- Ensure the LibTorch path is correctly set in CMakeLists.txt
- Make sure you have CMake and a C++ compiler with CUDA support installed
