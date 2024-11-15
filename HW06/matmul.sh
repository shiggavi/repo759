#!/bin/bash
#SBATCH --job-name=matmul 
#SBATCH --output="matmul.out"    
#SBATCH --error="matmul.err"      
#SBATCH --ntasks=1                   
#SBATCH --gpus-per-task=1            
#SBATCH --time=00:05:00              
#SBATCH --partition=instruction


module load nvidia/cuda/11.8.0

nvcc matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o matmul

EXECUTABLE=./matmul

# Define matrix size and threads per block
MATRIX_SIZE=1024
THREADS_PER_BLOCK=16

# Run the program
echo "Starting matmul job"
srun $EXECUTABLE $MATRIX_SIZE $THREADS_PER_BLOCK
echo "Matmul job completed"
