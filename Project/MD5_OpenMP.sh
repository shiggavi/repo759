#!/usr/bin/env zsh
#SBATCH --cpus-per-task=8              
#SBATCH --partition=instruction         
#SBATCH --job-name=MD5_OpenMP                
#SBATCH --output="MD5_OpenMP.out"            
#SBATCH --error="MD5_OpenMP.err"             
#SBATCH --time=0-00:45:00

cd $SLURM_SUBMIT_DIR 

# Compile the C++ code with OpenMP support
g++ MD5_OpenMP.cpp -Wall -O3 -std=c++17 -fopenmp -o MD5_OpenMP

for threads in $(seq 1 4); do
    ./MD5_OpenMP $threads
done