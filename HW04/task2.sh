#!/usr/bin/env zsh
#SBATCH --cpus-per-task=8
#SBATCH --partition=instruction
#SBATCH --job-name=Task2Slurm
#SBATCH --output="task2.out"
#SBATCH --error="task2.err"
#SBATCH --time=0-00:10:00

# Compile the task2.cpp code with optimization and OpenMP support
g++ task2.cpp -Wall -O3 -std=c++17 -o task2

# Run the simulation for a range of particles
for i in $(seq 100 100 1000);  # Runs for 100, 200, ..., 1000 particles
do 
    ./task2 $i 10.0  # Run task2 with the specified number of particles and 10.0 seconds of simulation
done
