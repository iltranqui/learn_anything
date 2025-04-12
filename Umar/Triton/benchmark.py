#!/usr/bin/env python3
"""
Comprehensive benchmark script for 3D convolution + GroupNorm + Mean operations.
This script can benchmark:
1. PyTorch Module implementation
2. PyTorch Functional implementation
3. Triton implementation (if available)

Results are saved to benchmark_results.txt and visualized in benchmark_results.png.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import os
import subprocess
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Check if Triton is available
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Triton not available. Only PyTorch benchmarks will be run.")

# Define the model
class ModelModule(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelModule, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        # Add noise to biases
        self.conv.bias = nn.Parameter(self.conv.bias + torch.ones_like(self.conv.bias) * 0.02)
        self.group_norm.bias = nn.Parameter(self.group_norm.bias + torch.ones_like(self.group_norm.bias) * 0.02)

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = x.mean(dim=[1, 2, 3, 4]) # Compute mean across all dimensions except batch
        return x

# Define the functional implementation
def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Applies 3D convolution, group normalization, and computes mean.
    """
    x = F.conv3d(x, conv_weight, bias=conv_bias)
    x = F.group_norm(x, num_groups, weight=group_norm_weight, bias=group_norm_bias)
    x = x.mean(dim=[1, 2, 3, 4])
    return x

# Define the functional model
class ModelFunctional(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    using functional API
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelFunctional, self).__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        group_norm = nn.GroupNorm(num_groups, out_channels)
        self.conv_weight = conv.weight
        self.conv_bias = nn.Parameter(
            conv.bias + torch.ones_like(conv.bias) * 0.02
        )
        self.group_norm_weight = group_norm.weight
        self.group_norm_bias = nn.Parameter(
            group_norm.bias + torch.ones_like(group_norm.bias) * 0.02
        )
        self.num_groups = num_groups

    def forward(self, x):
        return module_fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.group_norm_weight,
            self.group_norm_bias,
            self.num_groups,
        )

if HAS_TRITON:
    # Define a Triton kernel for the mean operation
    @triton.jit
    def mean_kernel(
        output_ptr, input_ptr,
        batch_size, channels, depth, height, width,
        stride_batch, stride_channel, stride_depth, stride_height, stride_width,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Get program ID
        pid = tl.program_id(0)
        
        # Each program handles one batch element
        if pid >= batch_size:
            return
            
        # Compute input pointers for this batch element
        input_batch_ptr = input_ptr + pid * stride_batch
        
        # Initialize sum
        sum_val = 0.0
        
        # Loop over all elements in this batch
        for c in range(channels):
            for d in range(depth):
                for h in range(height):
                    for w in range(width):
                        # Compute pointer to current element
                        ptr = input_batch_ptr + c * stride_channel + d * stride_depth + h * stride_height + w * stride_width
                        # Load value and add to sum
                        sum_val += tl.load(ptr)
        
        # Compute mean
        mean_val = sum_val / (channels * depth * height * width)
        
        # Store result
        tl.store(output_ptr + pid, mean_val)

    # Wrapper function for the Triton kernel
    def triton_mean(x):
        # Get input dimensions
        batch_size, channels, depth, height, width = x.shape
        
        # Create output tensor
        output = torch.empty((batch_size,), device=x.device, dtype=x.dtype)
        
        # Launch kernel
        grid = (batch_size,)
        mean_kernel[grid](
            output, x,
            batch_size, channels, depth, height, width,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            BLOCK_SIZE=128,
        )
        
        return output

    # Define a model that uses Triton for the mean operation
    class ModelTriton(nn.Module):
        """
        Model that performs a 3D convolution, applies Group Normalization, and uses Triton for mean
        """
        def __init__(self, in_channels, out_channels, kernel_size, num_groups):
            super(ModelTriton, self).__init__()
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
            self.group_norm = nn.GroupNorm(num_groups, out_channels)
            # Add noise to biases
            self.conv.bias = nn.Parameter(self.conv.bias + torch.ones_like(self.conv.bias) * 0.02)
            self.group_norm.bias = nn.Parameter(self.group_norm.bias + torch.ones_like(self.group_norm.bias) * 0.02)

        def forward(self, x):
            x = self.conv(x)
            x = self.group_norm(x)
            x = triton_mean(x)
            return x

def benchmark_model(model, x, num_iterations=100):
    """Benchmark a model with given input"""
    device = x.device
    
    # Warmup
    for _ in range(10):
        model(x)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(num_iterations):
        model(x)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    return avg_time

def run_cpp_benchmark():
    """Run the C++ benchmark and return the results"""
    # Check if the executable exists
    if not os.path.exists("build/conv3d_algorithm"):
        print("C++ benchmark executable not found. Building...")
        os.makedirs("build", exist_ok=True)
        subprocess.run(["cd build && cmake .. && make"], shell=True)
    
    # Run the benchmark
    subprocess.run(["./build/conv3d_algorithm"], shell=True)
    
    # Read the results
    if os.path.exists("libtorch_algorithm_results.txt"):
        with open("libtorch_algorithm_results.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "LibTorch implementation average time:" in line:
                    libtorch_time = float(line.split(":")[1].strip().split()[0])
                elif "Custom algorithm implementation average time:" in line:
                    custom_time = float(line.split(":")[1].strip().split()[0])
        
        return libtorch_time, custom_time
    else:
        print("C++ benchmark results not found.")
        return None, None

def plot_results(times, labels):
    """Plot benchmark results"""
    plt.figure(figsize=(12, 8))
    
    # Sort by time (ascending)
    sorted_indices = np.argsort([t for t in times])
    sorted_times = [times[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    
    # Use different colors for each bar
    colors = ['green', 'blue', 'orange', 'red', 'purple']
    colors = [colors[i % len(colors)] for i in range(len(sorted_times))]
    
    plt.bar(sorted_labels, sorted_times, color=colors)
    plt.ylabel('Time (ms)')
    plt.title('Performance Comparison of Different Implementations')
    plt.yscale('log')  # Use log scale for better visualization
    
    # Add time values on top of bars
    for i, v in enumerate(sorted_times):
        plt.text(i, v * 1.1, f"{v:.4f} ms", ha='center')
    
    # Add speedup annotations
    fastest_time = min(sorted_times)
    for i, time in enumerate(sorted_times):
        if i > 0:  # Skip the fastest implementation
            speedup = time / fastest_time
            plt.text(i, time / 2, f"{speedup:.1f}x slower", ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("Visualization saved to benchmark_results.png")

def main():
    # Parameters
    batch_size = 128
    in_channels = 3
    out_channels = 16
    D, H, W = 16, 32, 32
    kernel_size = 3
    num_groups = 8
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Create input tensor
    x = torch.randn(batch_size, in_channels, D, H, W, device=device)
    
    # Create models
    model_module = ModelModule(in_channels, out_channels, kernel_size, num_groups).to(device)
    model_functional = ModelFunctional(in_channels, out_channels, kernel_size, num_groups).to(device)
    
    # Results dictionary
    results = {}
    
    # Benchmark PyTorch implementations
    print("Benchmarking PyTorch Module implementation...")
    module_time = benchmark_model(model_module, x)
    results["PyTorch Module"] = module_time * 1000  # Convert to ms
    print(f"Module implementation average time: {module_time * 1000:.4f} ms")
    
    print("\nBenchmarking PyTorch Functional implementation...")
    functional_time = benchmark_model(model_functional, x)
    results["PyTorch Functional"] = functional_time * 1000  # Convert to ms
    print(f"Functional implementation average time: {functional_time * 1000:.4f} ms")
    
    # Benchmark Triton implementation if available
    if HAS_TRITON:
        model_triton = ModelTriton(in_channels, out_channels, kernel_size, num_groups).to(device)
        
        print("\nBenchmarking Triton implementation...")
        triton_time = benchmark_model(model_triton, x)
        results["Triton"] = triton_time * 1000  # Convert to ms
        print(f"Triton implementation average time: {triton_time * 1000:.4f} ms")
    
    # Run C++ benchmark
    print("\nRunning C++ benchmarks...")
    libtorch_time, custom_time = run_cpp_benchmark()
    
    if libtorch_time is not None:
        results["LibTorch (C++)"] = libtorch_time
        print(f"LibTorch implementation average time: {libtorch_time:.4f} ms")
    
    if custom_time is not None:
        results["Custom Algorithm (C++)"] = custom_time
        print(f"Custom algorithm implementation average time: {custom_time:.4f} ms")
    
    # Save results to file
    with open('benchmark_results.txt', 'w') as f:
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"CUDA version: {torch.version.cuda}\n\n")
        
        # Find the fastest implementation
        fastest_impl = min(results.items(), key=lambda x: x[1])
        fastest_time = fastest_impl[1]
        
        f.write("Results (sorted by speed):\n")
        for impl, time in sorted(results.items(), key=lambda x: x[1]):
            speedup = time / fastest_time
            speedup_str = "fastest (1.00x)" if impl == fastest_impl[0] else f"{speedup:.2f}x slower"
            f.write(f"{impl}: {time:.4f} ms - {speedup_str}\n")
    
    # Plot results
    plot_results(list(results.values()), list(results.keys()))
    
    print(f"\nResults saved to benchmark_results.txt")

if __name__ == "__main__":
    main()
