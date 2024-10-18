#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>  // For rand()
#include <omp.h>    // For OpenMP
#include "matmul.h"

using namespace std;
using namespace std::chrono;

// Function to generate a matrix of size n*n filled with random float values
void generate_matrix(float* matrix, unsigned int n) {
    for (unsigned int i = 0; i < n * n; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main(int argc, char* argv[]) {
    // Check for command line arguments
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " n t" << endl;
        return 1;
    }

    // Parse command line arguments
    unsigned int n = atoi(argv[1]);  // Matrix dimension
    unsigned int t = atoi(argv[2]);  // Number of threads

    // Set the number of threads for OpenMP
    omp_set_num_threads(t);

    // Seed random number generator
    srand(time(0));

    // Allocate memory for matrices A, B, and C
    float* A = new float[n * n];
    float* B = new float[n * n];
    float* C = new float[n * n];

    // Generate matrices A and B with random values
    generate_matrix(A, n);
    generate_matrix(B, n);

    // Measure time and run the parallel matrix multiplication
    auto start = high_resolution_clock::now();
    mmul(A, B, C, n);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    // Output the first and last element of C and the time taken
    cout << C[0] << endl;             // First element of C
    cout << C[n * n - 1] << endl;     // Last element of C
    cout << duration.count() << " ms" << endl;

    // Clean up memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}