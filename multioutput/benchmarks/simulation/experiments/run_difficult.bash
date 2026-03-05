#!/bin/bash
# Difficult: Weak Signal, High Noise, Strong Distractors
N_RUNS=15
N_JOBS=3

# Setup paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
BASE_OUTPUT_DIR="$SCRIPT_DIR/output/results_difficult"

mkdir -p "$BASE_OUTPUT_DIR"

# Run from project root to ensure module imports work
cd "$PROJECT_ROOT"

for EFFECT in "spike" "linear"; do
    echo "--- Running Difficult Benchmark ($EFFECT) ---"
    python3 -m code.simulation.main \
        --n_runs $N_RUNS --n_jobs $N_JOBS \
        --effect_type $EFFECT \
        --effect_magnitude 1.5 \
        --subject_noise 0.6 \
        --dispersion 2.0 \
        --nuisance_fraction 0.4 \
        --nuisance_amplitude 2.5 \
        --irregular_sampling_sd 3.0 \
        --output_dir "$BASE_OUTPUT_DIR/$EFFECT"
done
