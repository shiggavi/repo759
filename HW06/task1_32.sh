#! /usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=Task1
#SBATCH --output="task1_32.out"
#SBATCH --error="task1_32.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:05:00

# Load the CUDA module
module load nvidia/cuda/11.8.0

# Compile the CUDA program with specified flags
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

for i in $(seq 0 9);
do
  ./task1 $((2**(5+i))) 32
done

