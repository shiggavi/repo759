#!/usr/bin/python3
import matplotlib.pyplot as plt 

# Input file paths
file_path = "task3.out"       # Output from 512 threads per block
file_path1 = "task3_16.out"   # Output from 16 threads per block

y_pts = []  # Timing data for 512 threads
y_pts1 = [] # Timing data for 16 threads
x_pts = []  # Array size exponents (log2(n))

# Read results for 512 threads per block
with open(file_path, 'r') as file:
    for i, line in enumerate(file):
        if i % 3 == 0:  # Every 3rd line corresponds to kernel time
            y_pt = float(line.strip().split()[0])  # Take only the first value on the line
            y_pts.append(y_pt)
            x_pt = 10 + int(i / 3)  # Exponent (log2(n))
            x_pts.append(x_pt)

# Read results for 16 threads per block
with open(file_path1, 'r') as file:
    for i, line in enumerate(file):
        if i % 3 == 0 and line.strip():  # Every 3rd line and non-empty lines
            y_pt = float(line.strip().split()[0])  # Take only the first value on the line
            y_pts1.append(y_pt)

# Synchronize lengths
max_len = max(len(y_pts), len(y_pts1), len(x_pts))
while len(y_pts) < max_len:
    y_pts.append(0.0)  # Pad with 0 for missing data
while len(y_pts1) < max_len:
    y_pts1.append(0.0)  # Pad with 0 for missing data
while len(x_pts) < max_len:
    x_pts.append(10 + len(x_pts))  # Extend x_pts for extra data points

# Plot results
plt.figure(figsize=(10, 6))  # Set the figure size for better visibility
plt.xticks(x_pts, labels=[f'{i}' for i in x_pts], fontsize=10)
plt.yticks(fontsize=10)
plt.plot(x_pts, y_pts, label='512 Threads per Block', marker='o')
plt.plot(x_pts, y_pts1, label='16 Threads per Block', marker='x')

# Set labels, title, and legend
plt.title('Time Taken by vscale as a Function of Exponent (i)', fontsize=14)
plt.xlabel('Exponent (i) for n = $2^i$', fontsize=12)
plt.ylabel('Time Taken (ms)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot as a PDF
plt.savefig('task3.pdf')

