#!/bin/bash
# Easy: High Signal, Low Noise, No Distractors
N_RUNS=15
N_JOBS=3

# Setup paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
BASE_OUTPUT_DIR="$SCRIPT_DIR/output/results_easy"

mkdir -p "$BASE_OUTPUT_DIR"

# Run from project root to ensure module imports work
cd "$PROJECT_ROOT"

for EFFECT in "spike" "linear"; do
    echo "--- Running Easy Benchmark ($EFFECT) ---"
    python3 -m code.simulation.main \
        --n_runs $N_RUNS --n_jobs $N_JOBS \
        --n_time_points 10 \
        --effect_type $EFFECT \
        --effect_magnitude 4.0 \
        --subject_noise 0.1 \
        --dispersion 50.0 \
        --nuisance_fraction 0.0 \
        --irregular_sampling_sd 0.5 \
        --n_subjects 20 \
        --n_metabolites 100 \
        --output_dir "$BASE_OUTPUT_DIR/$EFFECT"
done
