#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include "msort.h"  // Make sure this file is in the same directory or the include path is set correctly

// Function to fill the array with random integers in the range [-1000, 1000]
void fillArray(int* arr, const std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = rand() % 2001 - 1000;  // Generates numbers between -1000 and 1000
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " n t ts\n";
        return 1;
    }

    // Parse command line arguments
    std::size_t n = std::stoul(argv[1]);
    int t = std::stoi(argv[2]);
    std::size_t ts = std::stoul(argv[3]);

    // Set number of threads
    omp_set_num_threads(t);

    // Create and fill the array with random numbers
    int* arr = new int[n];
    srand(time(0));  // Seed the random number generator
    fillArray(arr, n);

    // Record start time
    double start_time = omp_get_wtime();

    // Call the msort function to sort the array
    msort(arr, n, ts);

    // Record end time
    double end_time = omp_get_wtime();

    // Print the first and last elements of the sorted array
    std::cout << arr[0] << "\n";         // First element
    std::cout << arr[n - 1] << "\n";     // Last element

    // Print the time taken in milliseconds
    double time_taken = (end_time - start_time) * 1000;  // Convert to milliseconds
    std::cout << time_taken << "\n";

    // Clean up dynamically allocated memory
    delete[] arr;

    return 0;
}