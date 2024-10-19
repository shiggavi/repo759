#include "convolution.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <n> <t>\n";
        return 1;
    }

    std::size_t n = std::atoi(argv[1]);
    int t = std::atoi(argv[2]);

    // Check if the number of threads is valid
    if (t < 1 || t > 20)
    {
        std::cerr << "Error: t must be between 1 and 20.\n";
        return 1;
    }

    // Set the number of threads for OpenMP
    omp_set_num_threads(t);

    // Allocate and fill the image (n x n)
    float *image = new float[n * n];
    float *output = new float[n * n];

    // Create a 3x3 mask
    float mask[3 * 3] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Fill the image with random numbers in the range [-10, 10]
    for (std::size_t i = 0; i < n * n; ++i)
    {
        image[i] = static_cast<float>(rand()) / RAND_MAX * 20.0 - 10.0;
    }

    // Measure the time taken to apply the convolution
    high_resolution_clock::time_point start = high_resolution_clock::now();

    // Apply the convolution using the mask
    convolve(image, output, n, mask, 3);

    high_resolution_clock::time_point end = high_resolution_clock::now();
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    // Print the results
    cout << output[0] << "\n";                 // First element
    cout << output[n * n - 1] << "\n";         // Last element
    cout << duration.count() * 1000.0 << "\n"; // Time in milliseconds

    // Clean up
    delete[] image;
    delete[] output;

    return 0;
}