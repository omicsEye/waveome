#!/bin/sh

# One-time setup (login node, from examples/simulations). The environment is
# in the lab's shared space so anyone in the group can rerun the sweep, and
# requirements-pegasus.txt pins every dependency to the version it was built
# with. The library itself is installed as a snapshot of this checkout, not an
# editable link, so library edits cannot reach a sweep that is already
# running; re-run the last line after every pull that changes waveome/. (If
# it fails with Permission denied under build/, a stale build/ from an
# earlier install is in the way; it is git-ignored and safe to remove.)
#   module load gcc/12.2.0 python3/3.10.11
#   umask 022
#   python3 -m venv /GWSPH/groups/rahlab/venvs/waveome
#   . /GWSPH/groups/rahlab/venvs/waveome/bin/activate
#   pip install --upgrade pip
#   pip install -c requirements-pegasus.txt ../../. scikit-learn
#
# Submit the large cells (units * rate >= 1000) and the small cells
# separately, so each group can get its own resources (override any #SBATCH
# line below on the sbatch command line, e.g. -t or --mem). SLURM does not
# create the log directory, and a job whose log path does not exist fails
# with no output at all, so create it first:
#   . /GWSPH/groups/rahlab/venvs/waveome/bin/activate
#   mkdir -p logs
#   sbatch --array=1-$(python sim_waveome_hpc_run.py --size large --cells-per-task 4 --count-tasks) \
#       --export=ALL,SIZE=large,CELLS_PER_TASK=4 sim_waveome_hpc_script.sh
#   sbatch --array=1-$(python sim_waveome_hpc_run.py --size small --cells-per-task 12 --count-tasks) \
#       --export=ALL,SIZE=small,CELLS_PER_TASK=12 sim_waveome_hpc_script.sh
# Array indices are capped by the cluster's MaxArraySize (see
# `scontrol show config | grep -i MaxArraySize`). TASK_OFFSET is added to
# every index, so a longer range goes in chunks: tasks 1001-2000 are
#   sbatch --array=1-1000 --export=ALL,SIZE=...,CELLS_PER_TASK=...,TASK_OFFSET=1000 ...
# Finished cells are skipped, so resubmitting the same command resumes.

# Specify output files
#SBATCH -o ./logs/sim_waveome_%A_%a.out
#SBATCH -e ./logs/sim_waveome_%A_%a.err

# One process per array task; each GPSearch fit runs its 4 outcomes in parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Default partition: 14-day limit, nodes with 96 GB+ (list: sinfo -o "%P %l %c %m")
#SBATCH -p cpu

# Time limit (14 days)
#SBATCH -t 14-00:00:00

# Array ID: given on the sbatch command line (see above)

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

# Environment built once by the setup above
VENV=${VENV:-/GWSPH/groups/rahlab/venvs/waveome}
if [ ! -f "$VENV/bin/activate" ]; then
    echo "No environment at $VENV; run the one-time setup at the top of this script" >&2
    exit 1
fi
. "$VENV/bin/activate"

if [ -z "$SLURM_ARRAY_TASK_ID" ] || [ -z "$SIZE" ] || [ -z "$CELLS_PER_TASK" ]; then
    echo "Submit as an array with SIZE and CELLS_PER_TASK exported (see top of script)" >&2
    exit 1
fi

# One thread per Ray worker: the CPUs are already split across workers by
# --num-jobs (defaults to $SLURM_CPUS_PER_TASK)
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1

TASK_ID=$((SLURM_ARRAY_TASK_ID + ${TASK_OFFSET:-0}))
python sim_waveome_hpc_run.py $TASK_ID --size $SIZE --cells-per-task $CELLS_PER_TASK
