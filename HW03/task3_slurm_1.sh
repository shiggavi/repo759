#!/usr/bin/env zsh
#SBATCH --cpus-per-task=8
#SBATCH --partition=instruction
#SBATCH --job-name=Task3Slurm
#SBATCH --output="task3.out"
#SBATCH --error="task3.err"
#SBATCH --time=0-00:10:00


# Compile the C++ code
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp


# Define n = 106 and t = 8
n=106
t=8


# Run the task with ts = 2^1, 2^2, ..., 2^10
for ts in {1..10}; do
   ./task3 $n $t $((2 ** ts))
done