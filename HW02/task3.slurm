#!/bin/bash
#SBATCH -p instruction                 # Partition to submit to
#SBATCH -t 0-00:30:00                  # Maximum runtime (30 minutes)
#SBATCH --job-name=matmul_job          # Job name
#SBATCH --cpus-per-task=2              # Number of CPU cores to allocate
#SBATCH --output=matmul_job.out        # Output file for stdout
#SBATCH --error=matmul_job.err         # Error file for stderr

##cd $SLURM_SUBMIT_DIR


# Compile the code
g++ task3.cpp matmul.cpp -Wall -O3 -std=c++17 -o task3

# Run the program
./task3 1024

