#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()
#include "matmul.h"

using namespace std;
using namespace std::chrono;

// Function to generate a matrix of size n*n filled with random values
void generate_matrix(double* matrix, unsigned int n) {
    for (unsigned int i = 0; i < n * n; ++i) {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

// Function to generate a matrix stored as a std::vector<double>
std::vector<double> generate_matrix_vector(unsigned int n) {
    std::vector<double> matrix(n * n);
    for (unsigned int i = 0; i < n * n; ++i) {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    return matrix;
}

int main() {
    // Set matrix dimensions
    unsigned int n = 1024;  // At least 1000x1000 as per the requirements

    // Seed random number generator
    srand(time(0));

    // Allocate memory for matrices A, B, and C
    double* A = new double[n * n];
    double* B = new double[n * n];
    double* C = new double[n * n];

    // Generate matrices A and B with random values
    generate_matrix(A, n);
    generate_matrix(B, n);

    // Generate A and B as vectors for mmul4
    std::vector<double> A_vector = generate_matrix_vector(n);
    std::vector<double> B_vector = generate_matrix_vector(n);

    // Print the number of rows
    cout << n << endl;

    // Measure and run mmul1
    auto start = high_resolution_clock::now();
    mmul1(A, B, C, n);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << endl;
    cout << C[n * n - 1] << endl;  // Last element of C

    // Measure and run mmul2
    start = high_resolution_clock::now();
    mmul2(A, B, C, n);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << endl;
    cout << C[n * n - 1] << endl;

    // Measure and run mmul3
    start = high_resolution_clock::now();
    mmul3(A, B, C, n);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << endl;
    cout << C[n * n - 1] << endl;

    // Measure and run mmul4
    start = high_resolution_clock::now();
    mmul4(A_vector, B_vector, C, n);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << endl;
    cout << C[n * n - 1] << endl;

    // Clean up memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}  




