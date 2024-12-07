#! /usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=Task2_1024
#SBATCH --output="task2_1024.out"
#SBATCH --error="task2_1024.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00

module load nvidia/cuda/11.8.0
nvcc task2.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

for i in $(seq 10 30); do
  N=$((2**i))
  ./task2 $N 1024 
done


