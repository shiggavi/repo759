#! /usr/bin/env zsh
#SBATCH --partition=instruction          # Partition to run on
#SBATCH --job-name=Task3_16Threads       # Job name
#SBATCH --output="task3_16.out"          # Standard output file
#SBATCH --error="task3_16.err"           # Standard error file
#SBATCH --gres=gpu:1                     # Request one GPU
#SBATCH --time=0-00:05:00                # Maximum runtime of 5 minutes

# Load the CUDA module
module load nvidia/cuda/11.8.0

# Compile the CUDA program with specified flags
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

# Run task3 for n = 2^10 to 2^29 with 16 threads per block
for i in $(seq 1 20);
do
    ./task3 $((2**(9+i))) 16
done
