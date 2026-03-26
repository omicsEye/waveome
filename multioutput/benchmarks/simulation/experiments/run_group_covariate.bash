#!/bin/bash
# Group Covariate: Medium SNR with Binary Group Effect (50% attenuation in group 1)
N_RUNS=10
N_JOBS=1

# Setup paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."
BASE_OUTPUT_DIR="$SCRIPT_DIR/output/results_group_covariate"

if [[ "$*" == *"--overwrite"* ]]; then
    echo "Overwriting existing results..."
    rm -rf "$BASE_OUTPUT_DIR"
fi

mkdir -p "$BASE_OUTPUT_DIR"

# Run from project root to ensure module imports work
cd "$PROJECT_ROOT"

for EFFECT in "spike" "linear" "perturbation"; do
    echo "--- Running Group Covariate Benchmark ($EFFECT) ---"
    DONE=0
    while [ $DONE -lt $N_RUNS ]; do
        python3 -m simulation.main \
            --n_runs $N_RUNS --n_jobs $N_JOBS \
            --effect_type $EFFECT \
            --effect_magnitude 3.0 \
            --subject_noise 0.3 \
            --dispersion 10.0 \
            --nuisance_fraction 0.15 \
            --nuisance_amplitude 1.0 \
            --irregular_sampling_sd 1.5 \
            --n_subjects 20 \
            --n_metabolites 200 \
            --add_group_covariate \
            --condition_label "group_covariate" \
            --output_dir "$BASE_OUTPUT_DIR/$EFFECT"
        DONE=$(( $(wc -l < "$BASE_OUTPUT_DIR/$EFFECT/benchmark_results.csv" 2>/dev/null || echo 1) - 1 ))
        if [ $DONE -lt $N_RUNS ]; then
            echo "Process killed at $DONE/$N_RUNS, resuming in 5s..."
            sleep 5
        fi
    done
    echo "--- Completed $EFFECT ($DONE/$N_RUNS) ---"
done
