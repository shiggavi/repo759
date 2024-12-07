#! /usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --job-name=task1
#SBATCH --output="task1.out"
#SBATCH --error="task1.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00

# Load the necessary CUDA module
module load nvidia/cuda/11.8.0

# Compile the program
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

# Loop through matrix sizes and execute the task
for ((i=5; i<=14; i++));
do
    n=$((2**i))
    block_dim=16
    ./task1 $n $block_dim
done

