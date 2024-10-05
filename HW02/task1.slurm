#!/usr/bin/env zsh 
#SBATCH -p instruction
#SBATCH --job-name=FirstSlurm
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 0-00:30:00


cd $SLURM_SUBMIT_DIR

hostname

g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1

#running the program for n=2^10 to 2^30

for ((i=10; i<=30; i++)); do
n=$((2**i))
./task1 $n >> timings.txt
done

printf "%s\n" "${results[@]}" > plot.txt
