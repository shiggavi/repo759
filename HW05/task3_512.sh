#! /usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=FirstSlurm
#SBATCH --output="task3.out"
#SBATCH --error="task3.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:05:00

module load nvidia/cuda/11.8.0
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3
for i in $(seq 1 20);
do
        ./task3 $((2**(9+i)))
done
