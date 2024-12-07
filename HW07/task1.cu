#include <iostream>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include "matmul.cuh"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <block_dim>" << std::endl;
        return 1;
    }

    unsigned int n = std::atoi(argv[1]);
    unsigned int block_dim = std::atoi(argv[2]);

    // Allocate host memory
    int *h_Ai = new int[n * n];
    int *h_Bi = new int[n * n];
    float *h_Af = new float[n * n];
    float *h_Bf = new float[n * n];
    double *h_Ad = new double[n * n];
    double *h_Bd = new double[n * n];

    int *h_C1 = new int[n * n];
    float *h_C2 = new float[n * n];
    double *h_C3 = new double[n * n];

    // Fill matrices with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_int(-10, 10);
    std::uniform_real_distribution<float> dist_float(-10.0f, 10.0f);
    std::uniform_real_distribution<double> dist_double(-10.0, 10.0);

    for (unsigned int i = 0; i < n * n; ++i) {
        h_Ai[i] = dist_int(gen);
        h_Bi[i] = dist_int(gen);
        h_Af[i] = dist_float(gen);
        h_Bf[i] = dist_float(gen);
        h_Ad[i] = dist_double(gen);
        h_Bd[i] = dist_double(gen);
    }

    // Device memory allocation
    int *d_Ai, *d_Bi, *d_C1;
    float *d_Af, *d_Bf, *d_C2;
    double *d_Ad, *d_Bd, *d_C3;

    cudaMalloc(&d_Ai, n * n * sizeof(int));
    cudaMalloc(&d_Bi, n * n * sizeof(int));
    cudaMalloc(&d_C1, n * n * sizeof(int));
    cudaMalloc(&d_Af, n * n * sizeof(float));
    cudaMalloc(&d_Bf, n * n * sizeof(float));
    cudaMalloc(&d_C2, n * n * sizeof(float));
    cudaMalloc(&d_Ad, n * n * sizeof(double));
    cudaMalloc(&d_Bd, n * n * sizeof(double));
    cudaMalloc(&d_C3, n * n * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_Ai, h_Ai, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bi, h_Bi, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Af, h_Af, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bf, h_Bf, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ad, h_Ad, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bd, h_Bd, n * n * sizeof(double), cudaMemcpyHostToDevice);

    // Time each multiplication
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_1(d_Ai, d_Bi, d_C1, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms1;
    cudaEventElapsedTime(&ms1, start, stop);

    cudaEventRecord(start);
    matmul_2(d_Af, d_Bf, d_C2, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms2;
    cudaEventElapsedTime(&ms2, start, stop);

    cudaEventRecord(start);
    matmul_3(d_Ad, d_Bd, d_C3, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms3;
    cudaEventElapsedTime(&ms3, start, stop);

    // Copy results back
    cudaMemcpy(h_C1, d_C1, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C3, d_C3, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Output results
    std::cout << "matmul_1: " << ms1 << " ms, C[0]: " << h_C1[0] << ", C[n*n-1]: " << h_C1[n * n - 1] << std::endl;
    std::cout << "matmul_2: " << ms2 << " ms, C[0]: " << h_C2[0] << ", C[n*n-1]: " << h_C2[n * n - 1] << std::endl;
    std::cout << "matmul_3: " << ms3 << " ms, C[0]: " << h_C3[0] << ", C[n*n-1]: " << h_C3[n * n - 1] << std::endl;

    // Cleanup
    cudaFree(d_Ai);
    cudaFree(d_Bi);
    cudaFree(d_C1);
    cudaFree(d_Af);
    cudaFree(d_Bf);
    cudaFree(d_C2);
    cudaFree(d_Ad);
    cudaFree(d_Bd);
    cudaFree(d_C3);

    delete[] h_Ai;
    delete[] h_Bi;
    delete[] h_Af;
    delete[] h_Bf;
    delete[] h_Ad;
    delete[] h_Bd;
    delete[] h_C1;
    delete[] h_C2;
    delete[] h_C3;

    return 0;
}

