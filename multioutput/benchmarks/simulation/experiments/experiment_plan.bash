#!/bin/bash
# experiment_plan.bash — Sequential runner for the full benchmark suite.
#
# For local testing / small runs. For the actual HPC run use:
#   simulation/experiments/hpc/submit_all.sh
#
# Usage (from project root):
#   bash simulation/experiments/experiment_plan.bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
BASE_OUTPUT="$SCRIPT_DIR/output/final_benchmark"

# ── Configuration ─────────────────────────────────────────────────────────────
N_RUNS=50
N_JOBS=4
PYTHON="python3"   # override with full path if needed
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p "$BASE_OUTPUT"
cd "$PROJECT_ROOT"

# Medium-SNR base parameters reused across conditions
MED_SNR="--effect_magnitude 2.5 --subject_noise 0.3 --dispersion 10.0 --nuisance_fraction 0.2 --irregular_sampling_sd 1.5"
COMMON="--n_runs $N_RUNS --n_jobs $N_JOBS --skip_fitted_predictions"

run() {
    local label="$1" effect="$2" extra="$3" outdir="$4"
    echo "--- $label | effect=$effect ---"
    $PYTHON -m code.simulation.main $COMMON \
        --effect_type "$effect" \
        --condition_label "$label" \
        --output_dir "$BASE_OUTPUT/$outdir" \
        $extra
}

# ── Primary: Annotation Sweep ─────────────────────────────────────────────────
echo "=== Experiment 1: Annotation Sweep ==="
for ANNOT in 0.3 0.5 0.7 0.9; do
    for EFFECT in spike linear; do
        run "annot_${ANNOT}" "$EFFECT" \
            "$MED_SNR --annotation_fraction $ANNOT" \
            "annotation_sweep/annot_${ANNOT}/$EFFECT"
    done
done

# ── Secondary: SNR Sweep ──────────────────────────────────────────────────────
echo "=== Experiment 2: SNR Sweep ==="
for EFFECT in spike linear; do
    run "snr_easy" "$EFFECT" \
        "--effect_magnitude 4.0 --subject_noise 0.1 --dispersion 50.0 --nuisance_fraction 0.0 --irregular_sampling_sd 0.5 --n_time_points 10" \
        "snr_sweep/snr_easy/$EFFECT"

    run "snr_medium" "$EFFECT" \
        "$MED_SNR" \
        "snr_sweep/snr_medium/$EFFECT"

    run "snr_difficult" "$EFFECT" \
        "--effect_magnitude 1.5 --subject_noise 0.6 --dispersion 2.0 --nuisance_fraction 0.4 --irregular_sampling_sd 3.0" \
        "snr_sweep/snr_difficult/$EFFECT"
done

# ── Secondary: Group Covariate ────────────────────────────────────────────────
echo "=== Experiment 3: Group Covariate ==="
for EFFECT in spike linear; do
    run "group_covariate" "$EFFECT" \
        "$MED_SNR --effect_magnitude 3.0 --add_group_covariate" \
        "group_covariate/$EFFECT"
done

# ── Supplemental ──────────────────────────────────────────────────────────────
echo "=== Experiment 4: Supplemental Conditions ==="
for EFFECT in spike linear; do
    run "sparse_data" "$EFFECT" \
        "$MED_SNR --n_subjects 20 --n_time_points 3" \
        "supplemental/sparse_data/$EFFECT"

    run "nuisance_noise" "$EFFECT" \
        "$MED_SNR --nuisance_fraction 0.4 --nuisance_amplitude 2.5" \
        "supplemental/nuisance_noise/$EFFECT"

    run "high_irregularity" "$EFFECT" \
        "$MED_SNR --irregular_sampling_sd 4.0" \
        "supplemental/high_irregularity/$EFFECT"
done

echo ""
echo "=== All experiments complete ==="
echo "Results in: $BASE_OUTPUT"
echo "Next step: python3 simulation/experiments/aggregate_results.py $BASE_OUTPUT"
