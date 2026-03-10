#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1          # needed for MOGP (TensorFlow/GPflow). Remove if no GPU partition.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# run_condition.sbatch — SLURM job template for a single benchmark condition.
# Called by submit_all.sh; do not invoke directly.
#
# Required environment variables (set via --export in submit_all.sh):
#   CONDITION_LABEL   human-readable label written to benchmark_results.csv
#   EFFECT_TYPE       "spike" or "linear"
#   EXTRA_ARGS        additional CLI flags for this condition (quoted string)
#   OUTPUT_DIR        absolute path to output directory for this condition
#   PROJECT_ROOT      absolute path to the repository root
#   N_RUNS            number of simulation replicates
#   N_JOBS            parallel workers (should match --cpus-per-task)
#   PYTHON            path to python executable in the conda env
#   (R_HOME is set automatically by module load R/4.5.1 inside this script)

set -euo pipefail

# Load system R (required by rpy2 for comparison methods; adjust module name if needed)
module load R/4.5.1
export R_LIBS_USER=~/.R/library
# libRblas.so is bundled with R but not on the default library path
export LD_LIBRARY_PATH="/c1/apps/R/4.5.1/lib64/R/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== Job: $CONDITION_LABEL | effect=$EFFECT_TYPE ==="
echo "Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK  GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT/multioutput/benchmarks"

"$PYTHON" -m simulation.main \
    --n_runs "$N_RUNS" \
    --n_jobs "$N_JOBS" \
    --effect_type "$EFFECT_TYPE" \
    --condition_label "$CONDITION_LABEL" \
    --skip_fitted_predictions \
    --output_dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

echo ""
echo "=== Done: $CONDITION_LABEL | effect=$EFFECT_TYPE ==="
