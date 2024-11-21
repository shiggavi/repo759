#! /usr/bin/python3
import matplotlib.pyplot as plt 

file_path = "task1.out"
file_path1 = "task1_512.out"

y_pts = []
y_pts1 = []
x_pts = []

with open(file_path, 'r') as file:
    for i, line in enumerate(file):
        if(i ) % 2 == 1 :
            y_pt = float(line.strip())
            y_pts.append(y_pt)
            x_pt = 5+int(i/2)
            x_pts.append(x_pt)
with open(file_path1, 'r') as file:
    for i, line in enumerate(file):
        if(i ) % 2 == 1 :
            y_pt = float(line.strip())
            y_pts1.append(y_pt)

print(y_pts1)
plt.figure()
plt.xticks(x_pts, labels=[f'$2^{{{i}}}$' for i in x_pts])
plt.plot(x_pts , y_pts, label = 'threads_per_block = 1024', marker='o')
plt.plot(x_pts , y_pts1, label = 'threads_per_block = 32', marker='*')
plt.title('task1')
plt.xlabel('n')
plt.ylabel('matmul in ms')
plt.legend()
plt.savefig('task1.pdf')
