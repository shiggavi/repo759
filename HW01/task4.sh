#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --job-name=First Slurm
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time 00:01:00