#! /usr/bin/python3
import matplotlib.pyplot as plt

# File containing the output of task2
file_path = "task2_512.out"

# Lists to store x and y points
x_pts = []
y_pts = []

# Read the output file and extract data
with open(file_path, 'r') as file:
    for i, line in enumerate(file):
        # Extract time values on odd-indexed lines (1-based)
        if (i + 1) % 2 == 0:  # Assuming every second line contains the time
            y_pt = float(line.strip())  # Convert time value to float
            y_pts.append(y_pt)
            x_pt = 10 + i // 2  # Derive x-point based on the index (powers of 2 starting from 2^10)
            x_pts.append(x_pt)

# Generate the plot
plt.figure()
plt.xticks(x_pts, labels=[f'$2^{{{i}}}$' for i in x_pts])
plt.plot(x_pts, y_pts, label='threads_per_block = 512', marker='o')
plt.title('Reduction Performance')
plt.xlabel('Input Size (N)')
plt.ylabel('Time (ms)')
plt.legend()

# Save the plot as a PDF
plt.savefig('task2_512_plot.pdf')

