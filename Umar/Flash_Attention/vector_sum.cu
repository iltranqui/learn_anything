#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include "cuda_common.h"

type int EL_TYPE;


// __global__ is a CUDA specifier that indicates a function that runs on the device and can be called from the host
// CUDA can't know what each thread should do, so we need to tell it explicitly to CUDA -> we need to specify how the threads interact with each other
// CUDA launches threads in blocks of 32. If you require 43, threads, CUDA will launch 2 blocks per of total 64 threads, with 64-43=21 threads idle
// Group of 32 threads is called a warp which share a control unit

__global__ void cuda_vector_add_simple(EL_TYPE *OUT, EL_TYPE *A, EL_TYPE *B, int N) {
    int index = threadIdx.x; // threadIdx.x is a built-in variable that contains the index of the current thread in the block, each thread make a sum of single elements
    if (index < n) {
        OUT[index] = A[index] + B[index];
    }
}

// Main function for CUDA vector addition
// N: number of elements in the vector
void test_vector_add(int N)  {

    EL_TYPE *A, *B, *OUT;
    EL_TYPE *d_A, *d_B, *d_OUT;

    // Allocate the vectors on the host device -> the CPU
    A = (EL_TYPE *)malloc(sizeof(EL_TYPE) * N);
    B = (EL_TYPE *)malloc(sizeof(EL_TYPE) * N);
    OUT = (EL_TYPE *)malloc(sizeof(EL_TYPE) * N);

    // INitialize the vectors with random values
    for (int i = 0; i < N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Allocate device memory for a in the GPU
    CUDA_CHECK(cudaMalloc(void **)&d_A, sizeof(EL_TYPE) * N);
    CUDA_CHECK(cudaMalloc(void **)&d_B, sizeof(EL_TYPE) * N);
    CUDA_CHECK(cudaMalloc(void **)&d_OUT, sizeof(EL_TYPE) * N);

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(EL_TYPE) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeof(EL_TYPE) * N, cudaMemcpyHostToDevice));
    
    cudaEvent_t start_kernel, stop_kernel; 
    CUDA_CHECK(cudaEventCreate(&start_kernel));   // cudaEventCrate: This function creates an event object, which is used to record the timing of CUDA operations. 
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    // Launch the kernel
    CUDA_CHECK(cudaEventRecord(start_kernel)) // cudaEventRecord: This function records an event.
    // Run the kernel
    cuda_vector_add_simple<<<1 ,N>>>(d_OUT, d_A, d_B, N);   // launch N threads with the follwowingf arguments: d_OUT, d_A, d_B, N

}