#!/bin/sh

# One-time setup (login node, from examples/simulations). Installs a snapshot
# of the library, not an editable link, so library edits cannot reach a sweep
# that is already running; re-run the last line after changing waveome/.
#   module load gcc/12.2.0 python3/3.10.11
#   python3 -m venv $HOME/venvs/waveome
#   . $HOME/venvs/waveome/bin/activate
#   pip install --upgrade pip && pip install ../../. scikit-learn
#
# Submit the large cells (units * rate >= 1000) and the small cells
# separately, so each group can get its own resources (override any #SBATCH
# line below on the sbatch command line, e.g. -t or --mem). SLURM does not
# create the log directory, and a job whose log path does not exist fails
# with no output at all, so create it first:
#   . $HOME/venvs/waveome/bin/activate
#   mkdir -p logs
#   sbatch --array=1-$(python sim_waveome_hpc_run.py --size large --cells-per-task 4 --count-tasks) \
#       --export=ALL,SIZE=large,CELLS_PER_TASK=4 sim_waveome_hpc_script.sh
#   sbatch --array=1-$(python sim_waveome_hpc_run.py --size small --cells-per-task 12 --count-tasks) \
#       --export=ALL,SIZE=small,CELLS_PER_TASK=12 sim_waveome_hpc_script.sh
# Finished cells are skipped, so resubmitting the same command resumes.

# Specify output files
#SBATCH -o ./logs/sim_waveome_%A_%a.out
#SBATCH -e ./logs/sim_waveome_%A_%a.err

# One process per array task; each GPSearch fit runs its 4 outcomes in parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Use large memory queue
# SBATCH -p highMem
# SBATCH -p highThru
#SBATCH -p 384gb
# SBATCH -p defq

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
VENV=${VENV:-$HOME/venvs/waveome}
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

python sim_waveome_hpc_run.py $SLURM_ARRAY_TASK_ID --size $SIZE --cells-per-task $CELLS_PER_TASK
