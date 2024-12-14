#!/usr/bin/env zsh
#SBATCH --cpus-per-task=8              
#SBATCH --partition=instruction         
#SBATCH --job-name=MD5_Brute                
#SBATCH --output="MD5_Brute.out"            
#SBATCH --error="MD5_Brute.err"             
#SBATCH --time=0-00:45:00

cd $SLURM_SUBMIT_DIR

g++ MD5_Brute.cpp -Wall -O3 -std=c++17 -o MD5_Brute

./MD5_Brute
