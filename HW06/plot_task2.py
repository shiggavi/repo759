#! /usr/bin/python3
import matplotlib.pyplot as plt 

file_path = "task2.out"
file_path1 = "task2_512.out"

y_pts = []
y_pts1 = []
x_pts = []

with open(file_path, 'r') as file:
    for i, line in enumerate(file):
        if i % 2 == 1:
            values = line.strip().split()  # Split the line into individual numbers
            for value in values:
                y_pt = float(value)
                y_pts.append(y_pt)
                x_pt = 10 + int(i / 2)
                x_pts.append(x_pt)

with open(file_path1, 'r') as file:
    for i, line in enumerate(file):
        if i % 2 == 1:
            values = line.strip().split()  # Split the line into individual numbers
            for value in values:
                y_pt = float(value)
                y_pts1.append(y_pt)

print(y_pts1)
plt.figure()
plt.xticks(x_pts, labels=[f'$2^{{{i}}}$' for i in x_pts])
plt.plot(x_pts, y_pts, label='threads_per_block = 1024', marker='o')
plt.plot(x_pts, y_pts1, label='threads_per_block = 512', marker='x')
plt.title('task2')
plt.xlabel('n')
plt.ylabel('stencil in ms')
plt.legend()
plt.savefig('task2.pdf')

