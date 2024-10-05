#!/bin/bash

#SBATCH -p instruction                 # Partition to submit to
#SBATCH -t 0-00:30:00                  # Maximum runtime (30 minutes)
#SBATCH --job-name=convolution_job     # Job name
#SBATCH --cpus-per-task=2              # Number of CPU cores to allocate
#SBATCH --output=convolution_job.out   # Output file for stdout
#SBATCH --error=convolution_job.err    # Error file for stderr

cd $SLURM_SUBMIT_DIR

hostname

g++ convolution.cpp task2.cpp -Wall -O3 -std=c++17 -o task2

n=20   # Change n to the size you want for the image
m=17   # Change m to the size of the mask (must be an odd number)
./task2 $n $m
