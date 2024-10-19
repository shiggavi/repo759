
#! /usr/bin/env zsh
#SBATCH --cpus-per-task=20
#SBATCH --partition=instruction
#SBATCH --job-name=FirstSlurm
#SBATCH --output="task1.out"
#SBATCH --error="task1.err"
#SBATCH --cpus-per-task=20
#SBATCH --time=0-00:05:00


g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp
for i in $(seq 1 20);
do
      ./task1  1024 $((i))
done
