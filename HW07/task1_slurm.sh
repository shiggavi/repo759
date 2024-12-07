#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=MatrixMul
#SBATCH --output="task1.out"
#SBATCH --error="task1.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00

# Load the appropriate CUDA module
module load nvidia/cuda/11.8.0

# Compile the CUDA program
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

# Clean up the results file to prevent appending
rm -f results_q1.txt

# Run the program for n = 2^5 to 2^14 with block dimension 16
for i in $(seq 5 14);  # Ensure range ends at 14
do
  ./task1 $((2**i)) 16 >> results_q1.txt
done
