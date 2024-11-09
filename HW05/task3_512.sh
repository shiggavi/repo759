#!/bin/bash
#SBATCH --job-name=task3_512
#SBATCH --output="task3_512.out"
#SBATCH --error="task3_512.err"
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=instruction


module load nvidia/cuda/11.8.0

nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

./task3 512

