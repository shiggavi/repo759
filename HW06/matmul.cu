#include "matmul.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// Kernel function to compute the matrix product of A and B and store the result in C
__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    // Compute the row and column indices for this thread
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure row and column indices are within bounds
    if (row >= n || col >= n) {
        return;
    }

    // Compute the value of C[row][col]
    float value = 0.0f;
    for (size_t k = 0; k < n; ++k) {
        value += A[row * n + k] * B[k * n + col];
    }

    // Write the computed value to the output matrix
    C[row * n + col] = value;
}

// Host function to launch the matmul kernel
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    // Determine the block and grid dimensions
    dim3 threads_per_block_2d(threads_per_block, threads_per_block);
    dim3 num_blocks_2d((n + threads_per_block - 1) / threads_per_block, 
                       (n + threads_per_block - 1) / threads_per_block);

    // Launch the kernel
    matmul_kernel<<<num_blocks_2d, threads_per_block_2d>>>(A, B, C, n);

    // Wait for the kernel to finish execution
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

