#include "msort.h"
#include <algorithm>  // For std::swap
#include <omp.h>
#include <iostream>

// Function to merge two sorted subarrays into a sorted array
void merge(int* arr, int* tempArr, int left, int mid, int right) {
    int i = left;    // Starting index for left subarray
    int j = mid + 1; // Starting index for right subarray
    int k = left;    // Starting index to store merged subarray in tempArr

    // Merge the two subarrays
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            tempArr[k++] = arr[i++];
        } else {
            tempArr[k++] = arr[j++];
        }
    }

    // Copy remaining elements of left subarray, if any
    while (i <= mid) {
        tempArr[k++] = arr[i++];
    }

    // Copy remaining elements of right subarray, if any
    while (j <= right) {
        tempArr[k++] = arr[j++];
    }

    // Copy the sorted subarray back to the original array
    for (i = left; i <= right; i++) {
        arr[i] = tempArr[i];
    }
}

// Serial merge sort (used when array size is smaller than the threshold)
void serialMergeSort(int* arr, int* tempArr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        serialMergeSort(arr, tempArr, left, mid);
        serialMergeSort(arr, tempArr, mid + 1, right);
        merge(arr, tempArr, left, mid, right);
    }
}

// Parallel merge sort using OpenMP tasks
void parallelMergeSort(int* arr, int* tempArr, int left, int right, const std::size_t threshold) {
    std::size_t size = static_cast<std::size_t>(right - left + 1);  // Cast the difference to std::size_t
    if (size <= threshold) {
        // If the size of the array is below the threshold, use serial sort
        serialMergeSort(arr, tempArr, left, right);
    } else {
        int mid = left + (right - left) / 2;

        #pragma omp task shared(arr, tempArr) firstprivate(left, mid)
        parallelMergeSort(arr, tempArr, left, mid, threshold);

        #pragma omp task shared(arr, tempArr) firstprivate(mid, right)
        parallelMergeSort(arr, tempArr, mid + 1, right, threshold);

        #pragma omp taskwait  // Wait for both tasks to finish
        merge(arr, tempArr, left, mid, right);
    }
}

// Main merge sort function with OpenMP parallelization
void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    int* tempArr = new int[n]; // Temporary array for merging
    #pragma omp parallel
    {
        #pragma omp single
        parallelMergeSort(arr, tempArr, 0, n - 1, threshold);
    }
    delete[] tempArr;
}