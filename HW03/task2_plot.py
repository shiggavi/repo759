#!/usr/bin/python3
import matplotlib.pyplot as plt

file_path = "task2.out"

y_pts = []
x_pts = []

with open(file_path, 'r') as file:
    for i, line in enumerate(file):
        if (i) % 3 == 2:
            y_pt = float(line.strip())
            y_pts.append(y_pt)
            x_pt = int(i / 3) + 1
            x_pts.append(x_pt)

plt.figure()
plt.xticks(x_pts, labels=[str(label) for label in x_pts])
plt.plot(x_pts, y_pts, marker='*')
plt.title('task1')
plt.xlabel('number_of_threads')
plt.ylabel('mmul time in ms')

# Save as PDF
plt.savefig('task2.pdf')