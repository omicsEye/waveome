#!/bin/sh

# Specify output files
#SBATCH -o ihmp_waveome_%j.out
#SBATCH -e ihmp_waveome_%j.err

# Single node job
#SBATCH -N 1

# Use short queue
#SBATCH -p short

# Timelimit (4 hours)
#SBATCH -t 4:00:00

module load python3

python3 ihmp_waveome_hpc_run.py