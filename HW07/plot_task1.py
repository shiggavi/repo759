#! /usr/bin/python3
import matplotlib.pyplot as plt

# File containing the output of task1
file_path = "task1.out"

# Lists to store x and y points
x_pts = []
y_pts = []

# Read the output file and extract data
with open(file_path, 'r') as file:
    for i, line in enumerate(file):
        # Extract time values on odd-indexed lines (1-based)
        if (i + 1) % 3 == 0:  # Every third line corresponds to time
            y_pt = float(line.strip())
            y_pts.append(y_pt)
            x_pt = 5 + i // 3  # Derive x-point based on the index
            x_pts.append(x_pt)

# Generate the plot
plt.figure()
plt.xticks(x_pts, labels=[f'$2^{{{i}}}$' for i in x_pts])
plt.plot(x_pts, y_pts, label='threads_per_block = 1024', marker='o')
plt.title('Matrix Multiplication Performance')
plt.xlabel('Matrix Dimension (n)')
plt.ylabel('Time (ms)')
plt.legend()

# Save the plot as a PDF
plt.savefig('task1_plot.pdf')

