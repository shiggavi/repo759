#!/usr/bin/env zsh
#SBATCH --cpus-per-task=20
#SBATCH --partition=instruction
#SBATCH --job-name=FirstSlurm
#SBATCH --output="task2.out"
#SBATCH --error="task2.err"
#SBATCH --time=0-00:10:00

# Compile the C++ code
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

# Run the task with varying thread counts
for i in $(seq 1 20); do
    ./task2 1024 $i
done

python3 task2_plot.py


