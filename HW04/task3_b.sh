#!/usr/bin/env zsh
#SBATCH --cpus-per-task=8              # Maximum number of threads to use
#SBATCH --partition=instruction         # Partition name (adjust if needed)
#SBATCH --job-name=Task4Slurm           # Job name
#SBATCH --output="task4.out"            # Standard output file
#SBATCH --error="task4.err"             # Error output file
#SBATCH --time=0-00:30:00               # Maximum time limit

# Load modules (adjust based on Euler’s environment)
# module load gcc/10.2.0

# Compile the C++ code with OpenMP support
g++ task4.cpp -Wall -O3 -std=c++17 -fopenmp -o task4

# Create a results directory if it doesn't exist
mkdir -p results

# Run the simulation for different scheduling policies and thread counts
for policy in "static" "dynamic" "guided"; do
    echo "Running with $policy scheduling"
    export OMP_SCHEDULE=$policy  # Set OpenMP scheduling policy
    
    for threads in $(seq 1 8); do
        echo "Running with $threads threads"
        export OMP_NUM_THREADS=$threads  # Set the number of threads
        
        # Run the simulation for 100 particles and a simulation time of 100.0
        ./task4 100 100.0 $threads >> results/${policy}_results.txt
    done
done

# Generate plots using Python
python3 Task3_b.py
