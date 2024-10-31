import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_positions_from_csv(filename):
    """Read particle positions from CSV with the format: step, x1, y1, z1, x2, y2, z2, ..."""
    data = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            step = int(parts[0])  # Parse the step number
            
            # Parse positions as flat coordinates and reshape them
            coords = [float(val) for val in parts[1:]]
            positions = np.array(coords).reshape(-1, 3)  # Reshape to N x 3 matrix

            data.append(positions)

    return data

def main():
    filename = 'positions.csv'
    board_size = 4

    # Read particle positions from the CSV
    data = read_positions_from_csv(filename)

    # Prep figure
    fig = plt.figure(figsize=(8, 10), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:3, 0])

    for i in range(len(data)):
        pos = data[i]

        plt.sca(ax1)
        plt.cla()
        xx = np.array([p[0] for p in pos])
        yy = np.array([p[1] for p in pos])
        plt.scatter(xx, yy, s=1, color=[0.7, 0.7, 1])
        plt.scatter(pos[:, 0], pos[:, 1], s=10, color="blue")
        ax1.set(xlim=(-board_size, board_size), ylim=(-board_size, board_size))
        ax1.set_aspect("equal", "box")

        plt.pause(0.001)

    plt.savefig("nbody-cpp.png", dpi=240)
    plt.show()

if __name__ == "__main__":
    main()
