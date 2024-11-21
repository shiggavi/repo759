#! /usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=FirstSlurm
#SBATCH --output="task1.out"
#SBATCH --error="task1.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00

module load nvidia/cuda/11.8.0
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1
for i in $(seq 0 9);
do 	
  ./task1 $((2**(5+i))) 1024
done
