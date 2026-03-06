#!/bin/bash
# Group Covariate: Medium SNR with Binary Group Effect (50% attenuation in group 1)
N_RUNS=15
N_JOBS=3

# Setup paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
BASE_OUTPUT_DIR="$SCRIPT_DIR/output/results_group_covariate"

mkdir -p "$BASE_OUTPUT_DIR"

# Run from project root to ensure module imports work
cd "$PROJECT_ROOT"

for EFFECT in "spike" "linear"; do
    echo "--- Running Group Covariate Benchmark ($EFFECT) ---"
    python3 -m code.simulation.main \
        --n_runs $N_RUNS --n_jobs $N_JOBS \
        --effect_type $EFFECT \
        --effect_magnitude 3.0 \
        --subject_noise 0.3 \
        --dispersion 10.0 \
        --nuisance_fraction 0.2 \
        --nuisance_amplitude 1.0 \
        --irregular_sampling_sd 1.5 \
        --n_subjects 20 \
        --n_metabolites 100 \
        --add_group_covariate \
        --output_dir "$BASE_OUTPUT_DIR/$EFFECT"
done
