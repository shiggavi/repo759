#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

__global__
void factorial()
{
    int fact = 1;

    int index =threadIdx.x+1;

    for (int i = 1; i < index+1; i++)
    {
        fact *= i;
    }

    std::printf("%d!=%d \n", index, fact);
}

int main()
{
    factorial<<<1, 8>>>();

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
      return 1;
    }

    cudaDeviceSynchronize();

    return 0;
}


