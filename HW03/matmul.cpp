#include "matmul.h"
#include <omp.h>
#include <cstddef>

// Parallelized matrix multiplication using OpenMP
void mmul(const float* A, const float* B, float* C, const std::size_t n) {
    // Initialize the C matrix to zeros
    #pragma omp parallel for
    for (std::size_t i = 0; i < n * n; ++i) {
        C[i] = 0.0f;
    }

    // Matrix multiplication with parallelism in the outer loop
    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
