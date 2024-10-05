#include "matmul.h"

// Function mmul1: (i, j, k) loop order
void mmul1(const double* A, const double* B, double* C, const unsigned int n) {
    // Initialize matrix C to 0
    for (unsigned int i = 0; i < n * n; ++i) {
        C[i] = 0;
    }

    // Perform the matrix multiplication
    for (unsigned int i = 0; i < n; ++i) {       // loop over rows of C (and A)
        for (unsigned int j = 0; j < n; ++j) {   // loop over columns of C (and B)
            for (unsigned int k = 0; k < n; ++k) { // loop over the common dimension
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Function mmul2: (i, k, j) loop order
void mmul2(const double* A, const double* B, double* C, const unsigned int n) {
    // Initialize matrix C to 0
    for (unsigned int i = 0; i < n * n; ++i) {
        C[i] = 0;
    }

    // Perform the matrix multiplication
    for (unsigned int i = 0; i < n; ++i) {       // loop over rows of C (and A)
        for (unsigned int k = 0; k < n; ++k) {   // loop over columns of A and rows of B
            for (unsigned int j = 0; j < n; ++j) { // loop over columns of C (and B)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Function mmul3: (j, k, i) loop order
void mmul3(const double* A, const double* B, double* C, const unsigned int n) {
    // Initialize matrix C to 0
    for (unsigned int i = 0; i < n * n; ++i) {
        C[i] = 0;
    }

    // Perform the matrix multiplication
    for (unsigned int j = 0; j < n; ++j) {       // loop over columns of C (and B)
        for (unsigned int k = 0; k < n; ++k) {   // loop over the common dimension
            for (unsigned int i = 0; i < n; ++i) { // loop over rows of C (and A)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Function mmul4: (i, j, k) loop order but using std::vector for A and B
void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n) {
    // Initialize matrix C to 0
    for (unsigned int i = 0; i < n * n; ++i) {
        C[i] = 0;
    }

    // Perform the matrix multiplication
    for (unsigned int i = 0; i < n; ++i) {       // loop over rows of C (and A)
        for (unsigned int j = 0; j < n; ++j) {   // loop over columns of C (and B)
            for (unsigned int k = 0; k < n; ++k) { // loop over the common dimension
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
} 

