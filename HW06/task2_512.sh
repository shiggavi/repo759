#! /usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=FirstSlurm
#SBATCH --output="task2_512.out"
#SBATCH --error="task2_512.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00

module load nvidia/cuda/11.8.0
nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2
#./task2 32 1 4
for i in $(seq 0 19);
do 	
  ./task2 $((2**(10+i))) 128 512
done 


