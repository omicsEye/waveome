# Reproducibility Guide

This guide describes how to replicate the analysis environment for the `mogp-waveome` simulation framework.

## 1. System Requirements
- **OS**: macOS (tested on Apple Silicon) or Linux.
- **Conda**: [Miniforge](https://github.com/conda-forge/miniforge) is recommended.

> **macOS ARM64 note**: `environment.yml` uses `tensorflow-macos`. On Linux or Intel Mac,
> replace it with `tensorflow==2.15.0` in the pip section before creating the environment.

## 2. Environment Setup

### Step 1: Create the Conda Environment
Installs Python 3.11, R 4.5, rpy2, and all Python-based method dependencies.

```bash
conda env create -f environment.yml
conda activate mogp-waveome-sim
```

### Step 2: Install R Packages
Installs Bioconductor, CRAN, and GitHub R packages used by MEBA, PAL, timeOmics, and LMM methods.

```bash
Rscript install_r_deps.R
```

### Step 3: Set R_HOME (if needed)
If rpy2 cannot find R (e.g. when invoking Python directly rather than via an activated conda environment), set:

```bash
export R_HOME=$(conda run -n mogp-waveome-sim R RHOME)
```

## 3. Running the Benchmarks

```bash
conda activate mogp-waveome-sim

# Easy / medium / difficult SNR
bash simulation/experiments/run_easy.bash
bash simulation/experiments/run_medium.bash
bash simulation/experiments/run_difficult.bash

# Annotation quality sweep (annotation_fraction = 0.9, 0.5, 0.3)
bash simulation/experiments/run_annotation.bash

# Group covariate effect
bash simulation/experiments/run_group_covariate.bash
```
