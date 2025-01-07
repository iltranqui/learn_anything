#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include "cuda_common.h"

type int EL_TYPE;


// __global__ is a CUDA specifier that indicates a function that runs on the device and can be called from the host
// CUDA can't know what each thread should do, so we need to tell it explicitly to CUDA -> we need to specify how the threads interact with each other
// CUDA launches threads in blocks of 32. If you require 43, threads, CUDA will launch 2 blocks per of total 64 threads, with 64-43=21 threads idle
// Group of 32 threads is called ==  warp which share a control unit
// Control Unit: -> captain that orders tells a group of threads what to do ( workers )

__global__ void cuda_vector_add_simple(EL_TYPE *OUT, EL_TYPE *A, EL_TYPE *B, int N) {
    """
    Adding a simple vector addition kernel
    Parameters:
    OUT: output vector
    A: input vector A
    B: input vector B
    N: number of elements in the vector

    """
    int index = threadIdx.x; // threadIdx.x is a built-in variable that contains the index of the current thread in the block, each thread make a sum of single elements. THredID is like a passport for an execution
    if (index < n) {
        OUT[index] = A[index] + B[index];
    }
}

// Main function for CUDA vector addition
// N: number of elements in the vector
// Code written following video: 

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
    cuda_vector_add_simple<<<1 ,N>>>(d_OUT, d_A, d_B, N);   // launch N threads with the follwowing arguments: d_OUT, d_A, d_B, N
    CUDA_CHECK(cudaEventRecord(stop_kernel));  // cudaEventRecord: This function records, in this case the stop time of the kernel

    // Cuda for laucnh errors
    CUDA_CHECK(cudaPeekAtLastError());  // cudaPeekAtLastError: This function returns the last error that has been produced by any runtime API call.
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));  // cudaEventSynchronize: This function waits for an event to complete, to synchronize the CPU and other operations

    // Calculate the elapsed time
    float milliseconds_kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop_kernel));  // cudaEventElapsedTime: This function calculates the elapsed time between two events
    
}

void test_matrix_add(int NUM_ROWS, int NUM_COLS, int ROWS_block_size, int COLS_block_size)  {
    """
    Add two matrices using CUDA
    Parameters:
        NUM_ROWS: number of rows in the matrix
        NUM_COLS: number of columns in the matrix
        ROWS_block_size: number of blocks in the row dimension
        COLS_block_size: number of blocks in the column dimension
    """

    EL_TYPE *A, *B, *OUT;  // 
    EL_TYPE *d_A, *d_B, *d_OUT;

    // Allocate the matrices on the host device -> the GPU
    A = (EL_TYPE *)malloc(sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS);  // Allocate memory for the matrix A
    B = (EL_TYPE *)malloc(sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS);  // Allocate memory for the matrix B
    OUT = (EL_TYPE *)malloc(sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS);   // Allocate memory for the matrix OUT

    // Initialize the matrices with random values
    for (int i = 0; i < NUM_ROWS; i++) {
        for (int j = 0; j < NUM_COLS; j++) {
            size_t index = static_cast<size_t>(i) * NUM_COLS + j;
            A[index] = rand() % 100;
            B[index] = rand() % 100;
        }
    }

    // Allocate device memory for A, B, and OUT in the GPU
    CUDA_CHECK(cudaMalloc(void **)&d_A, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS);
    CUDA_CHECK(cudaMalloc(void **)&d_B, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS);
    CUDA_CHECK(cudaMalloc(void **)&d_OUT, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS);

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS, cudaMemcpyHostToDevice));

    cudaEvent_t start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    // Launch the kernel
    CUDA_CHECK(cudaEventRecord(start_kernel));

    // Define the launch grid
    int nums_blocks_rows = (NUM_ROWS + ROWS_block_size - 1) / ROWS_block_size;
    int nums_blocks_cols = (NUM_COLS + COLS_block_size - 1) / COLS_block_size;
    printf("Matrix Add - M: %d, N: %d, will be processed by (%d x %d) blocks of size (%d x %d)\n", NUM_ROWS, NUM_COLS, nums_blocks_rows, nums_blocks_cols, ROWS_block_size, COLS_block_size)
    dim3 grid(nums_blocks_rows, nums_blocks_cols, 1);
    dim3 block(ROWS_block_size, COLS_block_size, 1);
    // Run the kernel
    cuda_matrix_add<<<grid, block>>>(d_OUT, d_A, d_B, NUM_ROWS, NUM_COLS);

    // Cuda for launch errors
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));

    // Calculate the elapsed time
    float milliseconds_kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop_kernel));
    printf("Matrix Add - M: %d, N: %d, elapsed time: %f ms\n", NUM_ROWS, NUM_COLS, milliseconds_kernel);

    // Copy the result from device to host
    CUDA_CHECK(cudaMemcpy(OUT, d_OUT, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_OUT));

}