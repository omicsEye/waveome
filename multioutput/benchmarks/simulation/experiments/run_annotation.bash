#!/bin/bash
# Annotation Quality: Medium SNR across Annotation Coverage Levels (0.9, 0.5, 0.3)
N_RUNS=15
N_JOBS=3

# Setup paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
BASE_OUTPUT_DIR="$SCRIPT_DIR/output/results_annotation"

mkdir -p "$BASE_OUTPUT_DIR"

# Run from project root to ensure module imports work
cd "$PROJECT_ROOT"

for ANNOT in "0.9" "0.5" "0.3"; do
    for EFFECT in "spike" "linear"; do
        echo "--- Running Annotation Quality Benchmark (fraction=$ANNOT, $EFFECT) ---"
        python3 -m code.simulation.main \
            --n_runs $N_RUNS --n_jobs $N_JOBS \
            --effect_type $EFFECT \
            --effect_magnitude 2.5 \
            --subject_noise 0.3 \
            --dispersion 10.0 \
            --nuisance_fraction 0.2 \
            --nuisance_amplitude 1.0 \
            --irregular_sampling_sd 1.5 \
            --annotation_fraction $ANNOT \
            --output_dir "$BASE_OUTPUT_DIR/annot_${ANNOT}/$EFFECT"
    done
done
