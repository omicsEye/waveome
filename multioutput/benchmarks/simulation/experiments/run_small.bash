#!/bin/bash
# Small: Medium SNR, n_subjects=20, ALL methods including MEFISTO and timeOmics.
# Used for supplemental comparison showing MEFISTO/timeOmics performance at the
# scale where they are computationally tractable.
N_RUNS=10
N_JOBS=3

# Setup paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."
BASE_OUTPUT_DIR="$SCRIPT_DIR/output/results_small"

if [[ "$*" == *"--overwrite"* ]]; then
    echo "Overwriting existing results..."
    rm -rf "$BASE_OUTPUT_DIR"
fi

mkdir -p "$BASE_OUTPUT_DIR"

# Run from project root to ensure module imports work
cd "$PROJECT_ROOT"

for EFFECT in "spike" "linear" "perturbation"; do
    echo "--- Running Small Benchmark ($EFFECT, all methods, n_subjects=20) ---"
    python3 -m simulation.main \
        --n_runs $N_RUNS --n_jobs $N_JOBS \
        --effect_type $EFFECT \
        --effect_magnitude 4.0 \
        --subject_noise 0.5 \
        --dispersion 0.23 \
        --dispersion_spread 1.24 \
        --nuisance_fraction 0.15 \
        --nuisance_amplitude 1.0 \
        --irregular_sampling_sd 1.5 \
        --n_subjects 20 \
        --n_metabolites 200 \
        --method_timeout 2400 \
        --condition_label "snr_small_all_methods" \
        --output_dir "$BASE_OUTPUT_DIR/$EFFECT"
done
