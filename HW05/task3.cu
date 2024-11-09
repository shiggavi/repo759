#include <iostream>
#include <cuda.h>
#include <random>
#include "vscale.cuh"
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "Please provide the size of the array (n) as a command line argument." << endl;
        return 1;
    }

    unsigned int n = atoi(argv[1]);

    // Allocate memory for arrays on the host
    float* a = new float[n];
    float* b = new float[n];

    // Initialize random number generators
    default_random_engine gen;
    uniform_real_distribution<float> distribution1(-10.0, 10.0);
    uniform_real_distribution<float> distribution2(0.0, 1.0);

    for (unsigned int i = 0; i < n; i++) {
        a[i] = distribution1(gen);
        b[i] = distribution2(gen);
    }

    // Allocate memory on the device
    float* d_a;
    float* d_b;
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Set up the execution configuration
    dim3 blockDim(512);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

    // CUDA events for timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Start timing
    cudaEventRecord(start);
    
    // Launch the kernel
    vscale<<<gridDim, blockDim>>>(d_a, d_b, n);

    // End timing
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    cout << milliseconds << " ";

    // Copy the result back to host
    cudaMemcpy(b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the first and last element of the result
    cout << b[0] << " ";
    cout << b[n - 1] << endl;

    // Free memory
    delete[] a;
    delete[] b;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}

