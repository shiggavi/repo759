#!/bin/bash
#SBATCH --job-name=task2
#SBATCH --output="task2.out"
#SBATCH --error="task2.err"
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=instruction


module load nvidia/cuda/11.8.0

nvcc task2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

./task2

