#!/usr/bin/env zsh
#SBATCH --cpus-per-task=8
#SBATCH --partition=instruction
#SBATCH --job-name=Task3Slurm
#SBATCH --output="task3.out"
#SBATCH --error="task3.err"
#SBATCH --time=0-00:10:00

# Load necessary modules if required (adjust for your environment)
# module load gcc/10.2.0  # Example module, only if needed

# Compile the task3.cpp code with optimization and OpenMP support
g++ task3.cpp -Wall -O3 -std=c++17 -fopenmp -o task3

# Run the simulation for a range of particle counts
for i in $(seq 100 100 1000);  # Runs for 100, 200, ..., 1000 particles
do 
    echo "Running simulation with $i particles"
    ./task3 $i 10.0 8  # Run task3 with the specified number of particles, 10.0 seconds, and 8 threads
done

# Run the Python visualization script to generate PNG from CSV
echo "Generating PNG from CSV"
python3 plot_positions.py
