#!/bin/bash
# submit_all.sh — Submit all benchmark conditions as independent SLURM jobs.
#
# Usage (from project root):
#   bash simulation/experiments/hpc/submit_all.sh
#
# Each condition becomes one job. Jobs run in parallel on the cluster.
# Total wall time ~ wall time of the slowest single job, not the sum.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../../.." && pwd )"
SBATCH_TEMPLATE="$SCRIPT_DIR/run_condition.sbatch"
BASE_OUTPUT="$SCRIPT_DIR/../output/final_benchmark"

# ── Cluster configuration ────────────────────────────────────────────────────
export N_RUNS=50
export N_JOBS=32   # must match --cpus-per-task in run_condition.sbatch

# Fill in after running setup_hpc.sh:
CONDA_PREFIX=""    # e.g. /home/user/miniforge3/envs/mogp-waveome-sim
if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: Set CONDA_PREFIX to your conda env path before submitting."
    exit 1
fi
export PYTHON="$CONDA_PREFIX/bin/python3.11"
# R_HOME is set automatically by `module load R/4.5.1` inside run_condition.sbatch
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$PROJECT_ROOT/logs"

# Helper: submit one condition
# submit LABEL EFFECT EXTRA_ARGS OUTPUT_SUBDIR
submit() {
    local label="$1" effect="$2" extra="$3" subdir="$4"
    export CONDITION_LABEL="$label"
    export EFFECT_TYPE="$effect"
    export EXTRA_ARGS="$extra"
    export OUTPUT_DIR="$BASE_OUTPUT/$subdir"
    sbatch --job-name="${label}_${effect}" \
           --export=ALL \
           "$SBATCH_TEMPLATE"
    echo "  Submitted: ${label} | ${effect}"
}

# Medium-SNR base parameters used across all non-SNR-sweep conditions
MED_SNR="--effect_magnitude 2.5 --subject_noise 0.3 --dispersion 10.0 --nuisance_fraction 0.2 --irregular_sampling_sd 1.5"

echo "=== Submitting Primary: Annotation Sweep ==="
for ANNOT in 0.3 0.5 0.7 0.9; do
    for EFFECT in spike linear perturbation; do
        submit "annot_${ANNOT}" "$EFFECT" \
            "$MED_SNR --annotation_fraction $ANNOT" \
            "annotation_sweep/annot_${ANNOT}/${EFFECT}"
    done
done

echo ""
echo "=== Submitting Secondary: SNR Sweep ==="
for EFFECT in spike linear perturbation; do
    submit "snr_easy" "$EFFECT" \
        "--effect_magnitude 4.0 --subject_noise 0.1 --dispersion 50.0 --nuisance_fraction 0.0 --irregular_sampling_sd 0.5 --n_time_points 10" \
        "snr_sweep/snr_easy/${EFFECT}"

    submit "snr_medium" "$EFFECT" \
        "$MED_SNR" \
        "snr_sweep/snr_medium/${EFFECT}"

    submit "snr_difficult" "$EFFECT" \
        "--effect_magnitude 1.5 --subject_noise 0.6 --dispersion 2.0 --nuisance_fraction 0.4 --irregular_sampling_sd 3.0" \
        "snr_sweep/snr_difficult/${EFFECT}"
done

echo ""
echo "=== Submitting Secondary: Group Covariate ==="
for EFFECT in spike linear perturbation; do
    submit "group_covariate" "$EFFECT" \
        "$MED_SNR --effect_magnitude 3.0 --add_group_covariate" \
        "group_covariate/${EFFECT}"
done

echo ""
echo "=== Submitting Supplemental — Sparse Data / High Irregularity ==="
for EFFECT in spike linear perturbation; do
    submit "sparse_data" "$EFFECT" \
        "$MED_SNR --n_subjects 20 --n_time_points 3" \
        "supplemental/sparse_data/${EFFECT}"

    submit "high_irregularity" "$EFFECT" \
        "$MED_SNR --irregular_sampling_sd 4.0" \
        "supplemental/high_irregularity/${EFFECT}"
done

echo ""
echo "=== Submitting Supplemental — Temporal Resolution Sweep ==="
# Vary n_time_points at default n_subjects; tests whether GP temporal structure
# learning improves with more observations per unit.
for EFFECT in spike linear perturbation; do
    for N_TIME in 3 5 8 15; do
        submit "n_time_${N_TIME}" "$EFFECT" \
            "$MED_SNR --n_time_points $N_TIME" \
            "supplemental/n_time_sweep/n_time_${N_TIME}/${EFFECT}"
    done
done

echo ""
echo "=== Submitting Supplemental — Sample Size Sweep ==="
# Vary n_subjects at default n_time_points; tests how methods scale with
# the number of units (GP posterior precision vs. MEBA EB shrinkage vs. LMM stability).
for EFFECT in spike linear perturbation; do
    for N_SUBJ in 10 20 50 100; do
        submit "n_subj_${N_SUBJ}" "$EFFECT" \
            "$MED_SNR --n_subjects $N_SUBJ" \
            "supplemental/n_subj_sweep/n_subj_${N_SUBJ}/${EFFECT}"
    done
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor with: squeue -u \$USER"
echo "Results will land in: $BASE_OUTPUT"
