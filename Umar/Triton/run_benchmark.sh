#!/bin/bash

# Build the C++ implementation
echo "===== Building C++ Implementation ====="
mkdir -p build
cd build
cmake ..
make
cd ..

# Run the Python benchmark
echo -e "\n===== Running Python Benchmark ====="
python3 benchmark.py

echo -e "\n===== Benchmark Complete ====="
echo "Results saved to benchmark_results.txt"
echo "Visualization saved to benchmark_results.png"
