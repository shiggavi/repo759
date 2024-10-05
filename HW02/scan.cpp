#include "scan.h"

// Performs an inclusive scan on input array arr and stores
// the result in the output array
// arr and output are arrays of n elements
void scan(const float *arr, float *output, std::size_t n) {
    if (n == 0) {
        return; // No elements to process
    }
    
    // First element is the same as the input
    output[0] = arr[0];
    
    // Loop through the rest of the array and perform the inclusive scan
    for (std::size_t i = 1; i < n; ++i) {
        output[i] = output[i - 1] + arr[i];
    }
}
