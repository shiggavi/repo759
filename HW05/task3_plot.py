#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

# File paths
file_512 = "task3_512.out"
file_16 = "task3_16.out"

# Function to read the results
def read_results(file_path):
    sizes = []
    times = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):  # Every third line is the timing
            size = 2 ** (10 + (i // 3))
            # Extract only the first value on the line as the timing information
            time = float(lines[i].strip().split()[0])  # Split line and get the first item
            sizes.append(size)
            times.append(time)
    return sizes, times

# Read results for both configurations
sizes_512, times_512 = read_results(file_512)
sizes_16, times_16 = read_results(file_16)

# Create the plot
plt.figure()
plt.plot(sizes_512, times_512, marker='o', label="512 Threads per Block")
plt.plot(sizes_16, times_16, marker='x', label="16 Threads per Block")
plt.xscale('log')  # Logarithmic scale for sizes
plt.yscale('log')  # Logarithmic scale for times
plt.xlabel("Array Size (n)")
plt.ylabel("Time (ms)")
plt.title("vscale Execution Time vs Array Size")
plt.legend()

# Save the plot as a PDF
plt.savefig("task3.pdf")

