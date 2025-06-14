#!/bin/sh

# Specify output files
#SBATCH -o ./job_%A/sim_waveome_%a.out
#SBATCH -e ./job_%A/sim_waveome_%a.err

# Single-node job with 8 tasks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

# Use large memory queue
# SBATCH -p highMem
# SBATCH -p highThru
#SBATCH -p 384gb
# SBATCH -p defq

# Time limit (14 days)
#SBATCH -t 14-00:00:00

# Array ID
# SBATCH --array=1-10
#SBATCH --array=1-15

# Debug check
# SBATCH -p large-gpu
# SBATCH -t 4:00:00
# SBATCH --array=5

# Specify memory (64Gb)
# Comment out for now: SBATCH --mem=64000
# This always errors as well: SBATCH --mem-per-cpu=4GB

# Modules to load
module purge
module load gcc/12.2.0
module load python3/3.10.11
# module --ignore_cache load "gcc/12.2.0"
# module --ignore_cache load "python3/3.10.11"

# Packages to install
python3 pip install --upgrade --user pip
# python3 pip install --no-cache-dir tensorflow tensorflow_probability
python3 pip install --no-cache-dir --user ../../.
python3 pip install --user "ray[default]" statsmodels scikit-learn

# Now run script
python sim_waveome_hpc_run.py $SLURM_ARRAY_TASK_ID

