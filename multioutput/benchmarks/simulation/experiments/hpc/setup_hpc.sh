#!/bin/bash
# setup_hpc.sh — One-time environment setup on the HPC login node.
# Run this ONCE before submitting any jobs.
#
# Usage: bash simulation/experiments/hpc/setup_hpc.sh
#
# Prerequisites:
#   - conda/mamba available on PATH
#   - Internet access from login node (needed for R GitHub packages)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../../.."

# Initialize conda — required when running as a subprocess (bash script.sh
# does not inherit shell functions sourced in the parent session).
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/c1/apps/miniconda/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "=== Step 1: Create conda environment ==="
# Use mamba if available (faster), fall back to conda
CONDA_CMD="conda"
command -v mamba &>/dev/null && CONDA_CMD="mamba"

$CONDA_CMD env create -f "$PROJECT_ROOT/environment_linux.yml" --yes
echo "Conda environment 'mogp-waveome-sim' created."

echo ""
echo "=== Step 2: Install waveome from GitHub ==="
# conda env create cannot handle git+ URLs in some conda versions, so we pip install separately.
conda run -n mogp-waveome-sim pip install "git+https://github.com/omicsEye/waveome.git@multioutput_benchmarking"
echo "waveome installed."

echo ""
echo "=== Step 3: Install R dependencies ==="
# R packages must be installed from inside the conda env so they land in its R library.
# The R_HOME must point to the env's R, not the system R.
CONDA_PREFIX=$(conda run -n mogp-waveome-sim python -c "import sys; print(sys.prefix)")
R_HOME="$CONDA_PREFIX/lib/R"
R_BIN="$CONDA_PREFIX/bin/Rscript"

echo "Using R at: $R_BIN  (R_HOME=$R_HOME)"
R_HOME="$R_HOME" "$R_BIN" "$PROJECT_ROOT/install_r_deps.R"

echo ""
echo "=== Setup complete ==="
echo "Conda env path: $CONDA_PREFIX"
echo "To verify, run:"
echo "  conda activate mogp-waveome-sim"
echo "  python -c 'import waveome; import gpflow; print(\"OK\")'"
