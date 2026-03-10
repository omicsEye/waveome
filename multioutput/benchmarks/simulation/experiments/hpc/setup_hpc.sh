#!/bin/bash
# setup_hpc.sh — One-time environment setup on the HPC login node.
# Run this ONCE before submitting any jobs.
#
# Usage: bash simulation/experiments/hpc/setup_hpc.sh
#
# Prerequisites:
#   - conda available on PATH (source /c1/apps/miniconda/miniconda3/etc/profile.d/conda.sh)
#   - R/4.5.1 module loaded (module load R/4.5.1)
#   - Internet access from login node (needed for GitHub/PyPI packages)

set -euo pipefail

# ── Cluster-specific paths — adjust if your HPC differs ──────────────────────
R_MODULE="R/4.5.1"
R_HOME_PATH="/c1/apps/R/4.5.1/lib64/R"   # $(R RHOME) on the login node
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../../.."

# Initialize conda — required when running as a subprocess (bash script.sh
# does not inherit shell functions sourced in the parent session).
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/c1/apps/miniconda/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Load R module — conda's solver cannot handle r-base on this HPC's conda version.
# R is provided by the system module instead.
module load "$R_MODULE"

echo "=== Step 1: Create conda environment ==="
# Use python-only conda env; all other packages installed via pip to avoid
# conda solver compatibility issues with older conda versions on this HPC.
conda create -n mogp-waveome-sim -c conda-forge python=3.11 -y
echo "Conda environment 'mogp-waveome-sim' created."

echo ""
echo "=== Step 2: pip install all Python dependencies ==="
# conda run is unavailable in older conda; call pip via its full path instead.
CONDA_ENV_PATH=$(conda info --envs | grep "^mogp-waveome-sim " | awk '{print $NF}')

# rpy2 must be built against the system R. Set R_HOME, PATH, and linker flags
# so the build can find R and its internal BLAS (libRblas.so).
R_LIB_DIR="$R_HOME_PATH/lib"
export R_HOME="$R_HOME_PATH"
export PATH="$R_HOME_PATH/../bin:$PATH"
export LD_LIBRARY_PATH="$R_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LDFLAGS="-L$R_LIB_DIR"
"$CONDA_ENV_PATH/bin/pip" install rpy2==3.6.4
unset LDFLAGS  # only needed for rpy2 build; don't affect subsequent installs

# pywgcna==2.2.1 declares numpy>=2.1.0 but works fine on 1.26; install --no-deps
# after everything else so it doesn't force a numpy upgrade that breaks TF/GPflow.
"$CONDA_ENV_PATH/bin/pip" install \
  numpy==1.26.4 \
  gpflow==2.9.1 \
  tensorflow-probability==0.23.0 \
  tensorflow==2.15.0 \
  mofapy2==0.7.2 \
  muon==0.1.7 \
  anndata==0.12.7 \
  scanpy==1.11.5 \
  gseapy==1.1.11 \
  statsmodels==0.14.6 \
  scikit-learn==1.8.0 \
  joblib==1.5.3 \
  matplotlib==3.10.8 \
  seaborn==0.13.2 \
  tqdm==4.67.1 \
  "git+https://github.com/omicsEye/waveome.git@multioutput_benchmarking"
"$CONDA_ENV_PATH/bin/pip" install pywgcna==2.2.1 --no-deps
echo "Python dependencies installed."

echo ""
echo "=== Step 3: Install R package dependencies ==="
# R packages go to a user-writable library (system R library is read-only).
mkdir -p ~/.R/library
export R_LIBS_USER=~/.R/library
Rscript "$PROJECT_ROOT/multioutput/benchmarks/install_r_deps.R"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Conda env path (paste this into submit_all.sh as CONDA_PREFIX):"
echo "  CONDA_PREFIX=\"$CONDA_ENV_PATH\""
echo ""
echo "To verify the environment:"
echo "  module load $R_MODULE"
echo "  export R_LIBS_USER=~/.R/library"
echo "  export LD_LIBRARY_PATH=\"$R_LIB_DIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
echo "  $CONDA_ENV_PATH/bin/python3.11 -c 'import waveome; import gpflow; import rpy2; print(\"OK\")'"
