# HPC Simulation Workflow

Step-by-step guide for running the full benchmark suite on a SLURM cluster.

---

## Files in this directory

| File | Purpose |
|---|---|
| `setup_hpc.sh` | One-time environment setup on the login node |
| `run_condition.sbatch` | SLURM job template (one job per condition) |
| `submit_all.sh` | Master submission script — calls sbatch for all conditions |
| `HPC_WORKFLOW.md` | This file |

Supporting files referenced below:
- `environment_linux.yml` — conda env spec for reference (not used by setup_hpc.sh directly)
- `multioutput/benchmarks/install_r_deps.R` — R package installer
- `simulation/experiments/aggregate_results.py` — cross-condition plots and summary table
- `simulation/experiments/experiment_plan.bash` — sequential local runner (for testing)

---

## Order of operations

```
Local:   commit & push
HPC:     transfer code
HPC:     setup_hpc.sh          (once)
HPC:     [if needed] install waveome from source
HPC:     Test 1 — environment check
HPC:     Test 2 — MOGP timing at full scale
HPC:     Test 3 — full pipeline, 3 replicates
HPC:     edit submit_all.sh    (set CONDA_PREFIX)
HPC:     submit_all.sh         (20 parallel jobs)
HPC:     [wait for squeue to clear]
HPC:     aggregate_results.py
Local:   rsync paper_figures/ back
```

---

## Phase 1 — Prepare locally

Commit all changes so the repo is clean before transferring:

```bash
git add -A && git commit -m "HPC simulation setup"
git push origin multioutput_benchmarking
```

Note your waveome situation — see Phase 3b if the PyPI version is outdated.

---

## Phase 2 — Transfer code to HPC

**Option A (preferred) — git:**
```bash
# On the HPC login node
git clone <your-remote-url> <repo-dir>
cd <repo-dir>
git checkout multioutput_benchmarking
```

**Option B — rsync (if no GitHub remote):**
```bash
# From your local machine
rsync -avz \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
  /path/to/local/waveome/ \
  user@hpc.example.edu:<repo-dir>/
```

---

## Phase 3a — Environment setup

Run once from the project root on the HPC login node:

```bash
cd <repo-dir>
bash multioutput/benchmarks/simulation/experiments/hpc/setup_hpc.sh
```

This will:
1. Create a python-only `mogp-waveome-sim` conda environment (using conda-forge for Python 3.11)
2. Install all Python dependencies via pip (including waveome from the `multioutput_benchmarking` branch)
3. Install all R packages via `install_r_deps.R` into `~/.R/library`

**Prerequisites:** `module load R/4.5.1` and conda on PATH before running.

**Requires internet access from the login node.** Most HPC systems allow this. If yours
does not, the GitHub-sourced R packages (`MetaboAnalystR`, `PAL`, `PASI`, `lmms`) must
be installed from a compute node that has internet, or pre-bundled and installed from a
local path.

At the end of setup, the script prints the exact `CONDA_PREFIX` line to paste into `submit_all.sh` (Phase 5).

---

## Phase 3b — Updating waveome

`setup_hpc.sh` installs waveome directly from the `multioutput_benchmarking` branch on GitHub.
If you need to update it after the environment is set up (e.g., after pushing new commits):

```bash
CONDA_PREFIX=~/.conda/envs/mogp-waveome-sim   # adjust to actual path
$CONDA_PREFIX/bin/pip install --force-reinstall \
  "git+https://github.com/omicsEye/waveome.git@multioutput_benchmarking"
```

---

## Phase 4 — Test runs (do before submitting all jobs)

Set variables first:
```bash
module load R/4.5.1
export R_LIBS_USER=~/.R/library
export LD_LIBRARY_PATH="/c1/apps/R/4.5.1/lib64/R/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CONDA_PREFIX=~/.conda/envs/mogp-waveome-sim   # adjust to actual path
REPO=<path-to-waveome-repo>                   # e.g. /GWSPH/groups/rahlab/projects/waveome
```

### Test 1 — Environment check (< 2 min)

Verify Python imports and R connectivity:

```bash
$CONDA_PREFIX/bin/python3.11 -c "
import waveome, gpflow, rpy2
print('Python imports OK')
import rpy2.robjects as ro
ro.r('library(lme4); cat(\"R lme4 OK\n\")')
ro.r('library(PAL); cat(\"R PAL OK\n\")')
"
```

### Test 2 — MOGP timing (quick check before committing to 50 replicates)

Runs MOGP only (skips all other methods) for a single replicate at **reduced scale**
(n_subjects=20, n_metabolites=100) to get a fast environment check.
Should finish in 1–3 minutes on CPU:

```bash
cd "$REPO/multioutput/benchmarks"
$CONDA_PREFIX/bin/python3.11 -m simulation.main \
    --n_runs 1 --n_jobs 1 \
    --n_subjects 20 --n_metabolites 100 \
    --skip_wgcna --skip_mefisto --skip_dpgp --skip_timeomics \
    --skip_meba --skip_pal --skip_lmm \
    --skip_fitted_predictions \
    --output_dir /tmp/timing_test \
    --seed 42
```

Check `MOGP_Time` in `/tmp/timing_test/benchmark_results.csv`.

**Optional:** to estimate runtime at full scale (n_subjects=100, n_metabolites=500),
drop the `--n_subjects` / `--n_metabolites` flags. Full-scale MOGP on CPU takes ~15 min;
on GPU it should be < 5 min.

| MOGP_Time (full scale) | Action |
|---|---|
| < 5 min | Proceed as-is |
| 5–20 min | Fine — 50 replicates × 32 jobs is manageable within 48h |
| > 30 min | Verify GPU is being used (`nvidia-smi` during the run) |

### Test 3 — Full pipeline, 1 replicate at reduced scale (end-to-end check)

Runs all methods for one condition at **reduced scale** (n_subjects=20, n_metabolites=100)
to confirm the full output structure quickly (target: < 10 min total):

```bash
cd "$REPO/multioutput/benchmarks"
$CONDA_PREFIX/bin/python3.11 -m simulation.main \
    --n_runs 1 --n_jobs 1 \
    --n_subjects 20 --n_metabolites 100 \
    --effect_type spike \
    --annotation_fraction 0.7 \
    --effect_magnitude 2.5 --subject_noise 0.3 --dispersion 10.0 \
    --nuisance_fraction 0.2 --irregular_sampling_sd 1.5 \
    --condition_label "annot_0.7" \
    --skip_fitted_predictions \
    --output_dir /tmp/test_full \
    --seed 42
```

Check that `/tmp/test_full/benchmark_results.csv` contains `MOGP_GSEA_Sensitivity`,
`LMM_ORA_Sensitivity`, and all other expected columns before proceeding.

---

## Phase 5 — Configure and submit all jobs

**Edit `submit_all.sh`** — set the `CONDA_PREFIX` variable to the path found in Phase 3:

```bash
# Near the top of submit_all.sh:
CONDA_PREFIX=""    # e.g. /home/user/miniforge3/envs/mogp-waveome-sim
```

**Submit from the project root:**

```bash
cd ~/mogp-waveome
bash simulation/experiments/hpc/submit_all.sh
```

This submits ~20 independent SLURM jobs in one shot (annotation sweep × 8,
SNR sweep × 6, group covariate × 2, supplemental × 6). All run in parallel.

**Monitor:**
```bash
squeue -u $USER
# or for a live view:
watch -n 30 squeue -u $USER
```

**Job logs** land in `logs/` under the project root (one `.out` and `.err` per job).
Check them if any job fails:
```bash
ls logs/
tail -50 logs/annot_0.7_spike_<jobid>.out
```

If a job fails and needs to be rerun, just resubmit that condition manually:
```bash
export CONDITION_LABEL="annot_0.7"
export EFFECT_TYPE="spike"
export EXTRA_ARGS="--effect_magnitude 2.5 --subject_noise 0.3 --dispersion 10.0 --nuisance_fraction 0.2 --irregular_sampling_sd 1.5 --annotation_fraction 0.7"
export OUTPUT_DIR="$PWD/multioutput/benchmarks/simulation/experiments/output/final_benchmark/annotation_sweep/annot_0.7/spike"
export N_RUNS=50 N_JOBS=32
export PYTHON=$CONDA_PREFIX/bin/python3.11
export PROJECT_ROOT=$PWD
sbatch --job-name="annot_0.7_spike" --export=ALL \
    multioutput/benchmarks/simulation/experiments/hpc/run_condition.sbatch
```

---

## Phase 6 — Aggregate results

Once all jobs have finished (`squeue -u $USER` shows nothing):

```bash
cd "$REPO/multioutput/benchmarks"
module load R/4.5.1
export R_LIBS_USER=~/.R/library
export LD_LIBRARY_PATH="/c1/apps/R/4.5.1/lib64/R/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
$CONDA_PREFIX/bin/python3.11 \
    simulation/experiments/aggregate_results.py \
    simulation/experiments/output/final_benchmark \
    --out_dir paper_figures/
```

This produces in `paper_figures/`:
- `annotation_sweep_spike.png`, `annotation_sweep_linear.png` — primary figures
- `snr_sweep_spike.png`, `snr_sweep_linear.png` — supplemental
- `group_covariate_spike.png`, `group_covariate_linear.png` — supplemental
- `summary_table.csv` — mean ± SD for all methods and conditions

**Transfer figures back to local machine:**
```bash
# From your local machine
rsync -avz user@hpc.university.edu:~/mogp-waveome/paper_figures/ ./paper_figures/
```

---

## Experiment conditions submitted by submit_all.sh

### Primary — Annotation Sweep (main figures)
Base: medium SNR (effect_magnitude=2.5, subject_noise=0.3, dispersion=10.0, nuisance_fraction=0.2, irregular_sampling_sd=1.5)

| Condition | annotation_fraction | effect types | Jobs |
|---|---|---|---|
| `annot_0.3` | 0.3 | spike, linear | 2 |
| `annot_0.5` | 0.5 | spike, linear | 2 |
| `annot_0.7` | 0.7 | spike, linear | 2 |
| `annot_0.9` | 0.9 | spike, linear | 2 |

### Secondary — SNR Sweep (supplemental)

| Condition | effect_magnitude | subject_noise | dispersion | nuisance_fraction | irregular_sd | n_time_points |
|---|---|---|---|---|---|---|
| `snr_easy` | 4.0 | 0.1 | 50.0 | 0.0 | 0.5 | 10 |
| `snr_medium` | 2.5 | 0.3 | 10.0 | 0.2 | 1.5 | 5 |
| `snr_difficult` | 1.5 | 0.6 | 2.0 | 0.4 | 3.0 | 5 |

Each × spike + linear = 6 jobs.

### Secondary — Group Covariate (supplemental)
Medium SNR + `--add_group_covariate --effect_magnitude 3.0`, both effect types = 2 jobs.
Activates PAL and MEBA. MOGP includes `group` as a GP covariate automatically.

### Supplemental

| Condition | Key change |
|---|---|
| `sparse_data` | n_subjects=20, n_time_points=3 |
| `nuisance_noise` | nuisance_fraction=0.4, nuisance_amplitude=2.5 |
| `high_irregularity` | irregular_sampling_sd=4.0 |

Each × spike + linear = 6 jobs.

---

## Key simulation parameters (defaults as of 2026-03-04)

| Parameter | Default | Notes |
|---|---|---|
| `n_subjects` | 100 | |
| `n_metabolites` | 500 | 100 in pathways (20×5), 400 background |
| `n_pathways` | 5 | |
| `metabolites_per_pathway` | 20 | |
| `n_time_points` | 5 | |
| `annotation_fraction` | 0.7 | fraction of each pathway that is annotated |
| `n_runs` | 1 (CLI default) | set to 50 in submit_all.sh |
| `n_jobs` | 1 (CLI default) | set to 32 in submit_all.sh |

Active pathway is always `Pathway_1` (first pathway, metabolites M001–M020).
Group covariate: group 1 (50% of subjects) receives 50% of the temporal effect.

---

## Troubleshooting

**rpy2 cannot load libR.so / libRblas.so:** Ensure `module load R/4.5.1`, `export R_LIBS_USER=~/.R/library`, and `export LD_LIBRARY_PATH="/c1/apps/R/4.5.1/lib64/R/lib:$LD_LIBRARY_PATH"` are all set before running Python. `run_condition.sbatch` does this automatically for SLURM jobs.

**MOGP fails with TensorFlow/GPU errors:** Check `nvidia-smi` to confirm GPU is
allocated. Verify `CUDA_VISIBLE_DEVICES` is set by SLURM (`echo $CUDA_VISIBLE_DEVICES`
in the job). If no GPU, MOGP falls back to CPU but will be much slower.

**R GitHub package install blocked:** If the login node blocks outbound HTTPS, request
an interactive compute node with internet access, activate the conda env, and run
`Rscript install_r_deps.R` manually.

**A job finished but benchmark_results.csv is missing or empty:** Check the `.err` log
for that job. Common causes: import error (missing package), R package not installed,
or out-of-memory. Increase `--mem` in `run_condition.sbatch` if OOM.
