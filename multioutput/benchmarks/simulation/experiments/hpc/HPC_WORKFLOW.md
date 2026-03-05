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
- `environment_linux.yml` — conda env spec for Linux (project root)
- `install_r_deps.R` — R package installer
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
git push origin manuscript_comparative_methods
```

Note your waveome situation — see Phase 3b if the PyPI version is outdated.

---

## Phase 2 — Transfer code to HPC

**Option A (preferred) — git:**
```bash
# On the HPC login node
git clone <your-remote-url> ~/mogp-waveome
cd ~/mogp-waveome
git checkout manuscript_comparative_methods
```

**Option B — rsync (if no GitHub remote):**
```bash
# From your local machine
rsync -avz \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
  /Users/allen/Documents/Academics/GW/research/documents/manuscripts/mogp-waveome/ \
  user@hpc.university.edu:~/mogp-waveome/
```

---

## Phase 3a — Environment setup

Run once from the project root on the HPC login node:

```bash
cd ~/mogp-waveome
bash simulation/experiments/hpc/setup_hpc.sh
```

This will:
1. Create the `mogp-waveome-sim` conda environment from `environment_linux.yml`
2. Install all R packages via `install_r_deps.R`

**Requires internet access from the login node.** Most HPC systems allow this. If yours
does not, the GitHub-sourced R packages (`MetaboAnalystR`, `PAL`, `PASI`, `lmms`) must
be installed from a compute node that has internet, or pre-bundled and installed from a
local path.

At the end of setup, note the conda env path printed (e.g., `~/miniforge3/envs/mogp-waveome-sim`).
You will need it in Phase 5.

---

## Phase 3b — If waveome is not up to date on PyPI

The `environment_linux.yml` installs `waveome==0.1.2` from PyPI. If your latest MOGP
code lives in a local directory or a different git branch instead, do the following.

**1. Remove waveome from the yml** before running `setup_hpc.sh`:
```yaml
# In environment_linux.yml — delete or comment out:
# - waveome==0.1.2
```

**2. Transfer the waveome source to HPC:**
```bash
# Option A: rsync from local
rsync -avz /path/to/local/waveome/ user@hpc.university.edu:~/waveome/

# Option B: clone from repo
git clone <waveome-repo-url> ~/waveome
```

**3. After `setup_hpc.sh` finishes, install as editable package:**
```bash
CONDA_PREFIX=~/miniforge3/envs/mogp-waveome-sim   # adjust to actual path
$CONDA_PREFIX/bin/pip install -e ~/waveome/
```

The `-e` flag means Python uses the source directory directly — no reinstall needed
if you update the source later.

---

## Phase 4 — Test runs (do before submitting all jobs)

Set the conda prefix variable first:
```bash
CONDA_PREFIX=~/miniforge3/envs/mogp-waveome-sim   # adjust to actual path
```

### Test 1 — Environment check (< 2 min)

Verify Python imports and R connectivity:

```bash
R_HOME=$CONDA_PREFIX/lib/R $CONDA_PREFIX/bin/python3.11 -c "
import waveome, gpflow, rpy2
print('Python imports OK')
import rpy2.robjects as ro
ro.r('library(lme4); cat(\"R lme4 OK\n\")')
ro.r('library(PAL); cat(\"R PAL OK\n\")')
"
```

### Test 2 — MOGP timing at full scale (estimate runtime before committing to 50 replicates)

Runs MOGP only (skips all other methods) for a single replicate at the default
scale (n_subjects=100, n_metabolites=500):

```bash
cd ~/mogp-waveome
R_HOME=$CONDA_PREFIX/lib/R $CONDA_PREFIX/bin/python3.11 -m code.simulation.main \
    --n_runs 1 --n_jobs 1 \
    --skip_wgcna --skip_mefisto --skip_dpgp --skip_timeomics \
    --skip_meba --skip_pal --skip_lmm \
    --skip_fitted_predictions \
    --output_dir /tmp/timing_test \
    --seed 42
```

Check `MOGP_Time` in `/tmp/timing_test/benchmark_results.csv`:

| MOGP_Time | Action |
|---|---|
| < 5 min | Proceed as-is |
| 5–20 min | Fine — 50 replicates × 32 jobs is manageable within 48h |
| > 30 min | Reduce `--n_subjects` to 50, or verify GPU is being used (`nvidia-smi` during the run) |

### Test 3 — Full pipeline, 3 replicates (end-to-end check)

Runs all methods for one condition to confirm the full output structure:

```bash
cd ~/mogp-waveome
R_HOME=$CONDA_PREFIX/lib/R $CONDA_PREFIX/bin/python3.11 -m code.simulation.main \
    --n_runs 3 --n_jobs 3 \
    --effect_type spike \
    --annotation_fraction 0.7 \
    --effect_magnitude 2.5 --subject_noise 0.3 --dispersion 10.0 \
    --nuisance_fraction 0.2 --irregular_sampling_sd 1.5 \
    --condition_label "annot_0.7" \
    --skip_fitted_predictions \
    --output_dir /tmp/test_full \
    --seed 42
```

Check that `/tmp/test_full/benchmark_results.csv` contains `MOGP_ORA_Sensitivity`,
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
export OUTPUT_DIR="$PWD/simulation/experiments/output/final_benchmark/annotation_sweep/annot_0.7/spike"
export N_RUNS=50 N_JOBS=32
export PYTHON=$CONDA_PREFIX/bin/python3.11
export R_HOME=$CONDA_PREFIX/lib/R
sbatch --job-name="annot_0.7_spike" --export=ALL \
    simulation/experiments/hpc/run_condition.sbatch
```

---

## Phase 6 — Aggregate results

Once all jobs have finished (`squeue -u $USER` shows nothing):

```bash
cd ~/mogp-waveome
R_HOME=$CONDA_PREFIX/lib/R $CONDA_PREFIX/bin/python3.11 \
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

**R_HOME error from rpy2:** Ensure `R_HOME=$CONDA_PREFIX/lib/R` is set in every
command — rpy2 may point to the wrong R if this is not set explicitly.

**MOGP fails with TensorFlow/GPU errors:** Check `nvidia-smi` to confirm GPU is
allocated. Verify `CUDA_VISIBLE_DEVICES` is set by SLURM (`echo $CUDA_VISIBLE_DEVICES`
in the job). If no GPU, MOGP falls back to CPU but will be much slower.

**R GitHub package install blocked:** If the login node blocks outbound HTTPS, request
an interactive compute node with internet access, activate the conda env, and run
`Rscript install_r_deps.R` manually.

**A job finished but benchmark_results.csv is missing or empty:** Check the `.err` log
for that job. Common causes: import error (missing package), R package not installed,
or out-of-memory. Increase `--mem` in `run_condition.sbatch` if OOM.
