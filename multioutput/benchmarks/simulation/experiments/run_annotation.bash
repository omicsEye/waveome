#!/bin/bash
# Annotation Quality: Medium SNR across Annotation Coverage Levels (0.9, 0.5, 0.3)
N_RUNS=10
N_JOBS=2

# Setup paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."
BASE_OUTPUT_DIR="$SCRIPT_DIR/output/results_annotation"

if [[ "$*" == *"--overwrite"* ]]; then
    echo "Overwriting existing results..."
    rm -rf "$BASE_OUTPUT_DIR"
fi

mkdir -p "$BASE_OUTPUT_DIR"

# Run from project root to ensure module imports work
cd "$PROJECT_ROOT"

for ANNOT in "0.9" "0.5" "0.3"; do
    for EFFECT in "spike" "linear" "perturbation"; do
        echo "--- Running Annotation Quality Benchmark (fraction=$ANNOT, $EFFECT) ---"
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
            --n_subjects 100 \
            --n_metabolites 200 \
            --annotation_fraction $ANNOT \
            --condition_label "annot_${ANNOT}" \
            --output_dir "$BASE_OUTPUT_DIR/annot_${ANNOT}/$EFFECT"
    done
done
