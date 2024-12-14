#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --job-name=MD5_CUDA
#SBATCH -o MD5_CUDA.out
#SBATCH -e MD5_CUDA.err
#SBATCH --ntasks=1
#SBATCH --time=0-00:45:00
#SBATCH --gres=gpu:2

cd $SLURM_SUBMIT_DIR

module load nvidia/cuda/11.8.0

# Compile the CUDA program
nvcc MD5_CUDA.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o MD5_CUDA

# Run the compiled CUDA program
./MD5_CUDA