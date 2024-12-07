#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

# Input file path
file_path = "results_q1.txt"

# Initialize variables
x_pts = []  # Exponent values (log2(n))
y_matmul_1 = []  # Timing data for matmul_1
y_matmul_2 = []  # Timing data for matmul_2
y_matmul_3 = []  # Timing data for matmul_3

# Read and parse results file
with open(file_path, 'r') as file:
    lines = file.readlines()
    for i in range(0, len(lines), 3):  # Each group of 3 lines corresponds to one n value
        try:
            # Parse timing for matmul_1, matmul_2, matmul_3
            time_matmul_1 = float(lines[i].split(":")[1].split("ms")[0].strip())
            time_matmul_2 = float(lines[i + 1].split(":")[1].split("ms")[0].strip())
            time_matmul_3 = float(lines[i + 2].split(":")[1].split("ms")[0].strip())

            # Determine the exponent (log2(n))
            n_index = 5 + (i // 3)

            # Append to lists
            x_pts.append(n_index)
            y_matmul_1.append(time_matmul_1)
            y_matmul_2.append(time_matmul_2)
            y_matmul_3.append(time_matmul_3)
        except (ValueError, IndexError):
            continue

# Convert x_pts to actual matrix dimensions (2^i)
x_labels = [f'$2^{{{i}}}$' for i in x_pts]

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(x_pts, y_matmul_1, label='matmul_1 (int)', marker='o', color='r')
plt.plot(x_pts, y_matmul_2, label='matmul_2 (float)', marker='x', color='g')
plt.plot(x_pts, y_matmul_3, label='matmul_3 (double)', marker='s', color='b')

# Add labels, title, and grid
plt.title('Execution Time of Matrix Multiplication Functions vs Dimensions of Matrix', fontsize=16)
plt.xlabel('Dimension of Matrix ($2^i$)', fontsize=14)
plt.ylabel('Execution Time (ms)', fontsize=14)
plt.yscale('log')  # Logarithmic scale for the y-axis
plt.xticks(ticks=x_pts, labels=x_labels, fontsize=12, rotation=45)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Save and show the plot
plt.savefig('task1_plot_log.pdf')
plt.show()

