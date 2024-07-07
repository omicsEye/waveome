#!/bin/sh

# Specify output files
#SBATCH -o ihmp_waveome_%j.out
#SBATCH -e ihmp_waveome_%j.err

# Single node job
#SBATCH -N 1

# Use short queue
#SBATCH -p short

# Timelimit (12 hours)
#SBATCH -t 12:00:00

module load python3
python3 -m pip install ../../.
python3 ihmp_waveome_hpc_run.py

