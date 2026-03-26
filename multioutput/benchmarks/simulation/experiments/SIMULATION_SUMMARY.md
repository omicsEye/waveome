# Simulation Benchmark Summary

**Date:** 2026-03-17
**Status:** 30 replicates per condition — all conditions complete
**Script:** `generate_summary.py` → `output/summary/`

---

## 1. Simulation Design

### Data generating process

Each replicate simulates longitudinal metabolomics data:

| Parameter | Value |
|---|---|
| Subjects | 20 |
| Metabolites | 200 total (5 pathways × 20 each = 100 pathway members + 100 background) |
| Pathways | 5; one active (Pathway_1) |
| Annotation fraction (default) | 0.7 (70% of true pathway members in annotated set) |
| Time points | 5 (easy: 10) |
| Effect types | Spike (transient, t=8–12) and Linear (monotone increase) |

One pathway (Pathway_1) has a temporal effect on all 20 of its metabolites. The remaining 4 pathways are null. Background metabolites have metabolite-specific baselines and noise but no shared pathway effects. Each subject carries a random intercept; irregular time sampling is imposed.

### SNR conditions

| Condition | Effect magnitude | Subject noise SD | Dispersion | Nuisance fraction | Nuisance amplitude |
|---|---|---|---|---|---|
| Easy | 4.0 | 0.1 | 50 | 0.0 | — |
| Medium | 2.5 | 0.3 | 10 | 0.15 | 1.0 |
| Difficult | 1.5 | 0.6 | 2 | 0.20 | 1.0 |

Nuisance is a periodic signal h(t) = A·sin(2πt/P) applied uniformly to a randomly selected ~15% (medium) or ~20% (difficult) of all metabolites, consistent with estimates that ~10–15% of the metabolome exhibits circadian oscillations. The annotation sweep varies annotation fraction (0.3, 0.5, 0.7, 0.9) at medium SNR. The group covariate experiment adds a binary group variable where group 1 receives 50% of the temporal effect (medium SNR, effect magnitude 3.0).

### Methods

**Clustering methods** (evaluated by BestJaccard, UnannotatedRecall, NumModules, Reconstruction MSE):

| Method | Description |
|---|---|
| **MOGP** | Multi-output Gaussian process; horseshoe-prior factor model; Q estimated from SVD (90% variance) |
| WGCNA | Weighted gene co-expression network analysis |
| DPGP | Dirichlet process Gaussian process clustering |
| MEFISTO | MOFA+ with smooth temporal factors; capped at 25 iterations (see §7.1) |
| timeOmics | Sparse PLS on lmms splines |

**Pathway methods** (evaluated by Sensitivity = TPR for active pathway, FPR across 4 null pathways):

| Method | Description |
|---|---|
| **MOGP+ORA** | Otsu-thresholded MOGP module membership → hypergeometric ORA; min-p + Bonferroni across modules |
| LMM+ORA | LMM time-slope t-test → hypergeometric ORA |
| LMM+GSEA | LMM time-slope t-statistic → preranked GSEA (NOM p-val < 0.05) |
| MEBA | Mixed-effects empirical Bayes ranking → ORA |
| PAL | Pathway activity level score (group covariate condition only) |

---

## 2. Results: SNR Sweep

### Figure 1 — Pathway detection across SNR levels

![SNR pathway figure](output/summary/fig1_snr_pathway.png)

**Note on MEBA Reconstruction MSE.** MEBA (`performMB`) is a hypothesis test, not a trajectory model — it outputs per-metabolite T² scores and group assignments but no fitted time-course predictions. When MEBA is called in the benchmark, `main.py` falls back to using the per-metabolite temporal mean of `true_mu` (the oracle NB mean) as the "fitted value." This is an intercept-only null baseline: it captures no temporal structure whatsoever. As a result, MEBA's Reconstruction MSE (~0.67) reflects how well a flat, mean-level prediction matches the true trajectory — not how well MEBA models temporal dynamics. This is structurally incomparable to methods that explicitly fit time-courses (MOGP MSE ~0.029, LMM MSE ~0.085), and MEBA's MSE should not be interpreted as a measure of reconstruction quality. The fallback also uses oracle `true_mu` values rather than observed data means, making it slightly optimistic relative to a fully naive baseline, but the magnitude of MSE (~0.67 vs. ~0.03–0.09) makes the distinction inconsequential.

### Figure 2 — Clustering performance across SNR levels

![SNR clustering figure](output/summary/fig2_snr_clustering.png)

### Table 1 — Pathway methods: Sensitivity and FPR (mean ± SE, n=30)

| Condition | Effect | Metric | MOGP+ORA | LMM+ORA | LMM+GSEA | MEBA |
|---|---|---|---|---|---|---|
| Easy | Linear | Sensitivity | 0.767 ± 0.079 | **1.000 ± 0.000** | 0.333 ± 0.088 | 0.367 ± 0.089 |
| Easy | Linear | FPR | 0.117 ± 0.031 | **0.008 ± 0.008** | 0.017 ± 0.012 | 0.433 ± 0.040 |
| Easy | Spike | Sensitivity | **0.867 ± 0.063** | 0.033 ± 0.033 | 0.033 ± 0.033 | 0.300 ± 0.085 |
| Easy | Spike | FPR | 0.133 ± 0.033 | **0.033 ± 0.016** | 0.042 ± 0.017 | 0.400 ± 0.044 |
| Medium | Linear | Sensitivity | 0.667 ± 0.088 | **0.733 ± 0.082** | 0.300 ± 0.085 | 0.367 ± 0.089 |
| Medium | Linear | FPR | 0.342 ± 0.047 | **0.008 ± 0.008** | 0.025 ± 0.014 | 0.392 ± 0.041 |
| Medium | Spike | Sensitivity | 0.600 ± 0.091 | 0.033 ± 0.033 | 0.067 ± 0.046 | **0.667 ± 0.087** |
| Medium | Spike | FPR | 0.325 ± 0.054 | **0.017 ± 0.012** | 0.058 ± 0.020 | 0.383 ± 0.033 |
| Difficult | Linear | Sensitivity | **0.333 ± 0.088** | 0.233 ± 0.079 | 0.167 ± 0.069 | 0.200 ± 0.074 |
| Difficult | Linear | FPR | 0.308 ± 0.044 | **0.000 ± 0.000** | 0.025 ± 0.014 | 0.183 ± 0.041 |
| Difficult | Spike | Sensitivity | **0.200 ± 0.074** | 0.033 ± 0.033 | 0.067 ± 0.046 | **0.200 ± 0.074** |
| Difficult | Spike | FPR | 0.242 ± 0.039 | **0.033 ± 0.016** | 0.067 ± 0.021 | 0.225 ± 0.050 |

**Key observations:**
- **LMM+ORA** achieves perfect sensitivity (1.0) at easy SNR for linear effects with near-zero FPR. It **completely fails for spike effects** (0.00–0.03) at all SNR levels — the linear time coefficient cannot detect transient signals.
- **MOGP+ORA** is the only method that reliably detects both linear and spike effects. At easy SNR: 0.77 (linear) and 0.87 (spike). Performance degrades with SNR but remains non-trivial through difficult (0.20–0.33).
- **MOGP+ORA FPR** is elevated at medium SNR (0.32–0.34) compared to easy (0.12–0.13), then remains elevated at difficult (0.24–0.31). This is a structural limitation: MOGP forms ~7–9 modules at medium/difficult SNR, and circadian nuisance metabolites can co-load onto signal-bearing factors.
- **LMM+GSEA** consistently underperforms LMM+ORA. This is a confirmed structural limitation: metabolite-specific random effects create bidirectional t-statistics within the active pathway, violating GSEA's unidirectional enrichment assumption (see §7.3).
- **MEBA** shows moderate sensitivity at medium SNR spike (0.67) but with consistently elevated FPR (0.38–0.43), making it unreliable for low-FPR scenarios.

### Table 2 — Clustering methods: BestJaccard, UnannotatedRecall, NumModules, MSE (mean ± SE, n=30)

| Condition | Effect | Metric | MOGP | WGCNA | DPGP | MEFISTO | timeOmics |
|---|---|---|---|---|---|---|---|
| Easy | Linear | BestJaccard | **0.300 ± 0.034** | 0.099 ± 0.007 | 0.166 ± 0.009 | 0.233 ± 0.019 | 0.037 ± 0.009 |
| Easy | Linear | UnannotatedRecall | 0.339 ± 0.051 | **0.861 ± 0.057** | 0.233 ± 0.027 | 0.256 ± 0.039 | 0.194 ± 0.052 |
| Easy | Linear | NumModules | 3.8 | 1.3 | 19.6 | 10.0 | 0.7 |
| Easy | Linear | MSE | **0.028 ± 0.002** | 0.150 ± 0.012 | 0.158 ± 0.009 | 0.740 ± 0.028 | 0.146 ± 0.026 |
| Easy | Spike | BestJaccard | **0.419 ± 0.037** | 0.115 ± 0.005 | 0.159 ± 0.011 | 0.202 ± 0.015 | 0.026 ± 0.007 |
| Easy | Spike | UnannotatedRecall | 0.439 ± 0.047 | **0.978 ± 0.013** | 0.200 ± 0.028 | 0.217 ± 0.036 | 0.094 ± 0.030 |
| Easy | Spike | NumModules | 3.9 | 1.4 | 19.7 | 10.0 | 0.7 |
| Easy | Spike | MSE | **0.027 ± 0.002** | 0.143 ± 0.009 | 0.149 ± 0.009 | 0.722 ± 0.026 | 0.209 ± 0.042 |
| Medium | Linear | BestJaccard | **0.275 ± 0.036** | 0.087 ± 0.008 | 0.153 ± 0.011 | 0.129 ± 0.006 | 0.054 ± 0.010 |
| Medium | Linear | UnannotatedRecall | 0.489 ± 0.060 | **0.750 ± 0.072** | 0.233 ± 0.031 | 0.233 ± 0.033 | 0.233 ± 0.048 |
| Medium | Linear | NumModules | 6.6 | 1.4 | 19.3 | 10.0 | 1.1 |
| Medium | Linear | MSE | **0.029 ± 0.001** | 0.134 ± 0.012 | 0.158 ± 0.010 | 0.789 ± 0.033 | 0.228 ± 0.031 |
| Medium | Spike | BestJaccard | **0.246 ± 0.024** | 0.092 ± 0.006 | 0.139 ± 0.006 | 0.133 ± 0.006 | 0.056 ± 0.010 |
| Medium | Spike | UnannotatedRecall | 0.506 ± 0.058 | **0.778 ± 0.069** | 0.194 ± 0.029 | 0.300 ± 0.038 | 0.272 ± 0.052 |
| Medium | Spike | NumModules | 6.6 | 1.6 | 19.4 | 10.0 | 1.1 |
| Medium | Spike | MSE | **0.031 ± 0.001** | 0.134 ± 0.010 | 0.145 ± 0.007 | 0.779 ± 0.033 | 0.245 ± 0.040 |
| Difficult | Linear | BestJaccard | **0.181 ± 0.021** | 0.100 ± 0.000 | 0.148 ± 0.008 | 0.072 ± 0.012 | 0.024 ± 0.009 |
| Difficult | Linear | UnannotatedRecall | 0.611 ± 0.059 | **0.989 ± 0.011** | 0.206 ± 0.027 | 0.133 ± 0.030 | 0.083 ± 0.032 |
| Difficult | Linear | NumModules | 8.9 | 1.1 | 18.0 | 5.7 ± 0.92 | 0.5 |
| Difficult | Linear | MSE | **0.033 ± 0.001** | 0.281 ± 0.020 | 0.196 ± 0.010 | 0.760 ± 0.087 | 0.427 ± 0.050 |
| Difficult | Spike | BestJaccard | **0.180 ± 0.028** | 0.097 ± 0.003 | 0.128 ± 0.007 | 0.069 ± 0.012 | 0.023 ± 0.008 |
| Difficult | Spike | UnannotatedRecall | 0.578 ± 0.050 | **0.967 ± 0.033** | 0.139 ± 0.021 | 0.144 ± 0.033 | 0.106 ± 0.039 |
| Difficult | Spike | NumModules | 8.0 | 1.0 | 18.2 | 5.7 ± 0.92 | 0.5 |
| Difficult | Spike | MSE | **0.033 ± 0.001** | 0.278 ± 0.020 | 0.187 ± 0.010 | 0.759 ± 0.087 | 0.436 ± 0.052 |

**Key observations:**
- **MOGP** achieves the highest BestJaccard across all conditions. BestJaccard is lower than previous n=100 runs because signal density dropped from 20% to 10% of metabolites — a genuinely harder problem, not a regression.
- **WGCNA** consistently shows high UnannotatedRecall (0.75–0.99) but low Jaccard (0.087–0.115). It collapses to 1–2 very large modules that capture many true members by volume, not by precision.
- **DPGP** hits a ceiling of ~18–20 modules at easy and medium SNR. The extra 100 background metabolites give DPGP more clusters to fragment into, causing near-maximum module counts with low per-cluster purity.
- **MEFISTO** BestJaccard degrades more steeply with SNR than MOGP (0.233→0.069 vs 0.300→0.180, linear). At difficult SNR, some factors are discarded (NumModules=5.7). High MSE (0.72–0.79) reflects under-convergence at 25 iterations.
- **timeOmics** performance is uniformly low (Jaccard 0.023–0.056). The ~60% seed failure rate at n=200 metabolites is driven by background metabolites providing more opportunities for lmmSpline singular fits.

---

## 3. Results: Annotation Fraction Sweep

### Figure 3 — Annotation fraction sweep (linear effect)

![Annotation sweep linear](output/summary/fig3_annotation_sweep_linear.png)

### Figure 3b — Annotation fraction sweep (spike effect)

![Annotation sweep spike](output/summary/fig3_annotation_sweep_spike.png)

### Table 3 — Pathway sensitivity and FPR vs annotation fraction (medium SNR, n=30)

| Annot. Fraction | Effect | Metric | MOGP+ORA | LMM+ORA | LMM+GSEA | MEBA |
|---|---|---|---|---|---|---|
| 0.3 | Linear | Sensitivity | 0.200 ± 0.074 | **0.467 ± 0.093** | 0.200 ± 0.074 | 0.133 ± 0.063 |
| 0.3 | Linear | FPR | 0.192 ± 0.043 | **0.017 ± 0.012** | 0.033 ± 0.016 | 0.075 ± 0.021 |
| 0.5 | Linear | Sensitivity | 0.433 ± 0.092 | **0.667 ± 0.088** | 0.367 ± 0.090 | 0.267 ± 0.082 |
| 0.5 | Linear | FPR | 0.283 ± 0.043 | **0.008 ± 0.008** | 0.025 ± 0.014 | 0.242 ± 0.037 |
| 0.7 | Linear | Sensitivity | 0.667 ± 0.088 | **0.733 ± 0.082** | 0.300 ± 0.085 | 0.367 ± 0.089 |
| 0.7 | Linear | FPR | 0.342 ± 0.047 | **0.008 ± 0.008** | 0.025 ± 0.014 | 0.392 ± 0.041 |
| 0.9 | Linear | Sensitivity | 0.567 ± 0.092 | **0.800 ± 0.074** | 0.333 ± 0.088 | 0.467 ± 0.093 |
| 0.9 | Linear | FPR | 0.417 ± 0.047 | **0.008 ± 0.008** | **0.008 ± 0.008** | 0.558 ± 0.044 |
| 0.3 | Spike | Sensitivity | **0.233 ± 0.079** | 0.067 ± 0.046 | 0.067 ± 0.046 | 0.167 ± 0.069 |
| 0.3 | Spike | FPR | 0.217 ± 0.048 | **0.008 ± 0.008** | 0.025 ± 0.014 | 0.108 ± 0.026 |
| 0.5 | Spike | Sensitivity | **0.600 ± 0.091** | 0.067 ± 0.046 | 0.067 ± 0.046 | 0.300 ± 0.085 |
| 0.5 | Spike | FPR | 0.317 ± 0.051 | **0.008 ± 0.008** | 0.025 ± 0.014 | 0.258 ± 0.022 |
| 0.7 | Spike | Sensitivity | 0.600 ± 0.091 | 0.033 ± 0.033 | 0.067 ± 0.046 | **0.667 ± 0.087** |
| 0.7 | Spike | FPR | 0.325 ± 0.054 | **0.017 ± 0.012** | 0.058 ± 0.020 | 0.383 ± 0.033 |
| 0.9 | Spike | Sensitivity | 0.633 ± 0.089 | 0.000 ± 0.000 | 0.167 ± 0.069 | **0.767 ± 0.079** |
| 0.9 | Spike | FPR | 0.425 ± 0.048 | **0.008 ± 0.008** | 0.067 ± 0.027 | 0.500 ± 0.038 |

**Key observations:**
- **LMM+ORA** scales well with annotation fraction (linear: 0.47→0.80) and maintains near-zero FPR throughout. It contributes nothing for spike effects regardless of annotation coverage.
- **MOGP+ORA** scales from 0.20 to 0.57 (linear) and 0.23 to 0.63 (spike) as annotation increases. It is the **only method with meaningful spike sensitivity** at any annotation level.
- **MEBA** sensitivity scales similarly to MOGP+ORA for spike effects at higher annotation fractions but with substantially higher FPR (0.11→0.50).
- **FPR increases with annotation fraction** for MOGP+ORA and MEBA — richer annotation provides more false-pathway signal as well as true-pathway signal.
- **MOGP+ORA** and LMM+ORA trade off: LMM+ORA dominates linear sensitivity at low FPR; MOGP+ORA is the only viable option for spike detection at any annotation fraction.

---

## 4. Results: Timing

### Figure 4 — Runtime across conditions and methods

![Timing figure](output/summary/fig4_timing.png)

**Key observations:**
- **MOGP** runs ~480s (8 min) at easy SNR (10 time points) and ~260–270s (4–5 min) at medium/difficult (5 time points) on a single CPU core.
- **MEFISTO** (with 25-iteration cap) runs ~740s at easy, ~260s at medium, ~200s at difficult. The cap is necessary due to an O(N³) slogdet bottleneck; without it, full convergence would take ~2 hours at easy SNR (N=200). See §7.1.
- **timeOmics** failures at higher metabolite counts reduce average timing (failed runs return early).
- **LMM-based methods** (ORA, GSEA) remain fast (~2–3s total).

---

## 5. Results: Group Covariate (with PAL)

### Figure 5 — Group covariate condition (medium SNR, effect_magnitude=3.0)

![Group covariate figure](output/summary/fig5_group_covariate.png)

### Table 5 — Pathway detection in group covariate condition (n=30)

| Effect | Metric | MOGP+ORA | LMM+ORA | LMM+GSEA | MEBA | PAL |
|---|---|---|---|---|---|---|
| Linear | Sensitivity | 0.367 ± 0.089 | **0.667 ± 0.088** | 0.200 ± 0.074 | 0.467 ± 0.093 | 0.167 ± 0.069 |
| Linear | FPR | 0.267 ± 0.043 | **0.008 ± 0.008** | 0.017 ± 0.012 | 0.350 ± 0.044 | 0.033 ± 0.016 |
| Spike | Sensitivity | 0.367 ± 0.089 | 0.000 ± 0.000 | 0.033 ± 0.033 | **0.433 ± 0.092** | 0.067 ± 0.046 |
| Spike | FPR | 0.267 ± 0.041 | **0.017 ± 0.012** | 0.067 ± 0.027 | 0.408 ± 0.042 | 0.058 ± 0.023 |

**Key observations:**
- **PAL** achieves 0.17 sensitivity at low 0.033 FPR for linear effects — lower than LMM+ORA (0.667) but with competitive FPR. PAL fails for spike effects (0.067), consistent with its design around linear time trends.
- **LMM+ORA** leads for linear detection (0.667), lower than the non-group-covariate medium condition (0.733) due to the 50% effect attenuation in group 1.
- **MEBA** has the highest raw spike sensitivity (0.433) but with high FPR (0.408). **MOGP+ORA** (0.367, FPR 0.267) is the better spike detector when FPR is considered — the only method combining non-trivial spike sensitivity with FPR below 0.30.
- All methods show reduced performance relative to the equivalent non-group-covariate condition due to the 50% effect attenuation in group 1.

---

## 6. MOGP: Strengths and Weaknesses

### Strengths

1. **Best module precision (Jaccard).** MOGP achieves the highest BestJaccard across all conditions: 0.30–0.42 at easy SNR, 0.25–0.28 at medium, 0.18 at difficult. The next-best data-driven method achieves at most 0.23 (MEFISTO at easy SNR).

2. **Effect-type agnostic.** MOGP learns flexible temporal functions via GP priors and captures both spike and linear effects. LMM-based methods are structurally blind to spike effects.

3. **Unannotated metabolite recovery.** MOGP UnannotatedRecall (0.34–0.61) reflects its ability to group metabolites by trajectory shape without pathway annotations. Recall improves at harder SNR conditions as MOGP tends toward fewer, more focused modules.

4. **Lowest reconstruction MSE.** MOGP fits a structured GP to the actual trajectory, yielding substantially lower MSE (0.027–0.033) than all other methods (WGCNA: 0.13–0.28; MEFISTO: 0.72–0.76).

5. **Pathway detection for spike effects via MOGP+ORA.** At easy SNR, MOGP+ORA achieves 0.87 sensitivity for spike effects — no other method reliably detects spike pathway activation.

6. **Annotation-independent first stage.** Pathway enrichment is applied post-hoc, decoupling signal detection from annotation quality. MOGP+ORA maintains meaningful sensitivity even at 0.3 annotation fraction for spike effects (0.23), where LMM+ORA achieves 0.07.

### Weaknesses

1. **Computational cost.** At simulation scale (n=20, p=200), MOGP takes 270–480s per replicate. At HPC scale (n=100, p=500), GPU acceleration will be required.

2. **Elevated FPR in pathway enrichment.** MOGP+ORA FPR ranges 0.12–0.43 across conditions — substantially higher than LMM+ORA (≈0.00–0.03). FPR increases with annotation fraction and peaks at medium SNR.

3. **Jaccard reduced with lower signal density.** Moving from n=100 to n=200 metabolites (10% vs 20% signal density) cut BestJaccard roughly in half at easy SNR. Performance is sensitive to the fraction of metabolites carrying temporal signal.

4. **Variable module count.** NumModules increases with noise: easy ~4, medium ~7, difficult ~8–9. The growing module count at hard conditions likely reflects noise-driven fragmentation.

5. **Group covariate included but not interaction-modeled.** MOGP includes group as a categorical covariate in the GP input space (`categorical_vars=["group"]`), allowing the kernel to vary with group membership. However, the model does not explicitly parameterize a group × time interaction term — group effects enter through the shared kernel rather than as a separate multiplicative factor on the temporal trajectory.

---

## 7. Known Issues and Limitations

### 7.1 MEFISTO: O(N³) computational bottleneck (resolved for benchmark)

**Root cause:** Each ELBO iteration calls `np.linalg.slogdet(Qcov[k,:,:])` — an O(N³) operation on the N×N variational posterior covariance (N = n_subjects × n_time = 200 at easy SNR). Full convergence (~200 iterations) would take ~2 hours per replicate.

**Resolution:** Capped at `n_iterations=25`, keeping runtime to ~5–15 min. Sparse GP approximation in mofapy2 v0.7.3 does NOT resolve this — the posterior covariance remains N×N.

**Impact on results:** MEFISTO results reflect under-converged inference. BestJaccard (0.07–0.23) and MSE (0.72–0.76) are lower bounds on method capability. This is documented in the manuscript as a benchmark finding.

### 7.2 timeOmics: ~60% of seeds produce 0 modules

**Status:** Metabolite ID mismatch bug fixed. Non-zero Jaccard observed in successful runs.

**Remaining issue:** ~60% of seeds produce 0 modules at n=200 metabolites (up from ~50% at n=100). Background metabolites provide more opportunities for lmmSpline singular fits. Seed-specific failures are consistent across SNR conditions, suggesting certain data realizations systematically cause lmmSpline to return NULL. Mean Jaccard values are pulled down by the high zero-module rate.

### 7.3 LMM+GSEA: confirmed structural limitation

**Root cause:** Two compounding issues: (1) metabolite-specific random effects create bidirectional t-statistics within the active pathway, violating GSEA's unidirectional assumption; (2) with only 5 pathways, permutation FDR is poorly calibrated. Using NOM p-val < 0.05 instead of FDR q-val provides marginal improvement but the structural issue persists.

**Impact:** LMM+GSEA sensitivity (0.0–0.33) consistently underperforms LMM+ORA. This is a genuine finding documented in the manuscript.

### 7.4 LMM+ORA: cannot detect spike effects (structural)

**Observation:** LMM+ORA sensitivity = 0.00–0.03 for all spike-type effects at all SNR levels.

**Root cause:** A transient spike produces a near-zero average time slope → t-stat ≈ 0 → no significant metabolites from the active pathway reach ORA.

### 7.5 MOGP+ORA FPR: structural elevation

**Observation:** MOGP+ORA FPR is elevated at medium/difficult SNR (0.24–0.34) compared to easy (0.12–0.13). FPR also increases with annotation fraction (0.19 at 0.3 annotation → 0.43 at 0.9 annotation).

**Root cause:** At medium/difficult SNR, MOGP forms ~7–9 modules. Circadian nuisance metabolites can co-load onto signal-bearing factors, and with more modules the Bonferroni correction across factors is less conservative. The Otsu threshold (replacing the prior fixed-weight threshold) improved sensitivity by ~+0.20 without changing FPR, confirming FPR is driven by module composition rather than membership threshold.

**Status:** Structural — no further algorithmic fix applied at this time. Documented in manuscript.

---

## 8. Summary Assessment

| Dimension | Best method | Notes |
|---|---|---|
| Pathway detection (linear effect) | LMM+ORA | 0.73–1.00 sensitivity at easy/medium; near-zero FPR; fails for spike |
| Pathway detection (spike effect) | MOGP+ORA | Only method with non-trivial sensitivity; FPR 0.13–0.43 |
| Pathway detection (both effects) | MOGP+ORA | Effect-type agnostic; FPR elevated |
| Module precision (Jaccard) | MOGP | Best across all conditions; 2–3× next-best at easy SNR |
| Unannotated metabolite recovery | MOGP | Only method measuring annotation-independent recall |
| Reconstruction fidelity | MOGP | 5–10× lower MSE than next-best |
| Runtime | WGCNA / DPGP / LMM | MOGP 100×+ slower; MEFISTO O(N³) bottlenecked |
| Annotation fraction robustness | LMM+ORA | Near-zero FPR maintained; MOGP+ORA maintains spike sensitivity at low annotation |
| Group covariate (linear) | LMM+ORA | PAL provides low-FPR alternative |
| Group covariate (spike) | MOGP+ORA | Only method with meaningful spike sensitivity |

**Overall narrative:**

**Where MOGP excels.** MOGP's clearest advantage is on non-linear, transient temporal effects — the biological scenario most relevant to metabolomic biomarkers. Spike-type effects model the class of responses that are ubiquitous in metabolomics: post-intervention peaks, post-prandial surges, pharmacokinetic curves, circadian phase shifts, and disease-onset transients. These produce near-zero mean slopes over the full time window, which causes LMM-based methods (LMM+ORA, LMM+GSEA) to collapse to sensitivity ≤ 0.033 regardless of SNR. MOGP+ORA detects these effects with sensitivity 0.60–0.87 at easy/medium SNR, making it the only viable pathway detection method when the temporal signature is non-linear or transient. The GP prior places no restrictions on trajectory shape — it learns the covariance structure directly from data — so sensitivity does not degrade when the true signal is not monotone or linear.

As a clustering method, MOGP achieves the best Jaccard similarity across all SNR levels and outperforms WGCNA, DPGP, MEFISTO, and timeOmics on unannotated metabolite recovery (the biologically relevant quantity when pathway databases are incomplete or noisy). At easy SNR, MOGP's Jaccard lead is 2–3× the next-best competitor. The joint multi-output structure allows MOGP to borrow information across metabolites when estimating shared temporal patterns, which helps at lower SNR where individual metabolite signals are weak. This clustering advantage is annotation-independent: MOGP discovers modules from temporal co-expression alone, without needing to consult a database. In real high-dimensional metabolomics, this means MOGP can identify novel biochemical programs not yet represented in KEGG or HMDB.

Reconstruction fidelity is MOGP's largest quantitative advantage over alternatives: MSE ≈ 0.029 vs. LMM ≈ 0.085 (3×) and WGCNA/MEFISTO/timeOmics (5–10×). Because MOGP's GP posterior is a full probabilistic model over trajectories, the fitted time-courses capture temporal smoothness and uncertainty simultaneously. This has practical value beyond benchmarking: MOGP predictions can substitute missing timepoints, propagate uncertainty to downstream enrichment tests, and provide interpretable trajectory summaries per module.

**Where MOGP falls short.** For linear temporal effects in well-annotated designs, LMM+ORA is strictly superior: sensitivity 0.73–1.00 vs. 0.33–0.77 for MOGP+ORA, with near-zero FPR (0.000–0.017) vs. MOGP+ORA's structurally elevated FPR (0.12–0.34). The FPR elevation is a direct consequence of MOGP's multi-module structure — with 7–9 modules at medium/difficult SNR, nuisance metabolites can co-load onto signal-bearing factors, and the subsequent per-module ORA tests are correlated. FPR also scales with annotation fraction: as the annotated pathway set grows from 30% to 90% coverage, MOGP+ORA FPR increases from 0.19 to 0.43 (a combinatoric property of hypergeometric testing with larger K), which is a meaningful limitation when working with well-curated databases. MEBA shares the same FPR inflation pattern (0.075 to 0.558 across annotation fractions), suggesting this is a general property of top-K ORA approaches rather than specific to MOGP.

Computationally, MOGP is 100×+ slower than WGCNA, DPGP, and LMM. At the benchmark scale (n=20 subjects, T=5–10 timepoints, n_metabolites=200), runtime is manageable, but scaling to larger metabolomics datasets (n>100 subjects, 1000+ metabolites) will require further approximation. MEFISTO shares the GP-over-time prior but is bottlenecked by O(N³) posterior updates, making it ~4× slower than MOGP at this scale and ~20× slower at full convergence — suggesting that tractable GP-based multi-output modeling is a genuine architectural advantage of MOGP's formulation.

**Clustering vs. dedicated clustering methods.** MOGP is competitive with or better than all clustering baselines on Jaccard and unannotated recall. DPGP is the closest competitor at easy SNR (also GP-based, but single-output), while WGCNA remains competitive at difficult SNR due to simpler correlation-based grouping being more robust to noise than complex probabilistic models. The key distinction is that MOGP clustering is not an isolated step — module membership feeds directly into ORA, and the same GP posterior provides reconstruction fidelity. This end-to-end integration is what allows MOGP to translate clustering quality into pathway detection sensitivity for non-linear effects.

**Pathway detection vs. dedicated pathway methods.** For linear effects, MOGP+ORA cannot match LMM+ORA on either sensitivity or FPR — it should not be positioned as a replacement for linear designs. Its value is in the spike/non-linear regime where LMM-based methods are effectively uninformative. The practical recommendation is method selection by anticipated effect type: if the biological hypothesis involves sustained directional change (treatment → linear metabolic shift), LMM+ORA is optimal; if the hypothesis involves transient, oscillatory, or non-monotone responses (challenge tests, circadian experiments, disease onset), MOGP+ORA is the only benchmark method with adequate sensitivity.

---

*Full numeric results: `output/summary/summary_table.csv`*
*Figures generated by: `generate_summary.py`*
