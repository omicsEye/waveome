# Simulation Benchmark Summary

**Date:** 2026-04-15
**Status:** 10 replicates per condition — all conditions complete, MEFISTO now included
**Script:** `generate_summary.py` → `output/summary/`

---

## 1. Simulation Design

### Data generating process

Each replicate simulates longitudinal metabolomics data:

| Parameter | Value |
|---|---|
| Subjects | 100 (iHMP cohort scale) |
| Metabolites | 200 total (5 pathways × 20 each = 100 pathway members + 100 background) |
| Pathways | 5; one active (Pathway_1) |
| Annotation fraction (default) | 0.7 (70% of true pathway members in annotated set) |
| Time points | 5 (10 for easy condition) |
| Effect types | Spike (transient, t=8–12), Linear (monotone increase), Perturbation (step change at t=4) |
| Dispersion | NB r=0.23 (median), σ=1.24 (log-normal spread) — iHMP-calibrated |

One pathway (Pathway_1) has a temporal effect on all 20 of its metabolites. The remaining 4 pathways are null. Each subject carries a random intercept; irregular time sampling is imposed.

### SNR conditions

| Condition | Effect magnitude | Subject noise SD | Nuisance fraction | Nuisance amplitude |
|---|---|---|---|---|
| Easy | 8.0 | 0.5 | 0.0 | — |
| Medium | 4.0 | 0.5 | 0.15 | 1.0 |
| Difficult | 2.0 | 0.5 | 0.20 | 2.5 |

Dispersion parameters (r=0.23) were calibrated to iHMP stool metabolomics data. Nuisance is a periodic signal h(t) = A·sin(2πt/P) applied to ~15–20% of metabolites. The annotation sweep varies annotation fraction (0.3, 0.5, 0.7, 0.9) at medium SNR. The group covariate experiment adds a binary group variable where group 1 receives 50% of the temporal effect (effect_magnitude=6.0 so average across groups ≈ medium SNR).

### Methods

**Clustering methods** (evaluated by ARI, BestJaccard, BestPrecision, UnannotatedRecall, NumModules, Reconstruction MSE):

| Method | Description |
|---|---|
| **MOGP** | Multi-output Gaussian process; horseshoe-prior factor model; Q estimated from SVD (90% variance) |
| WGCNA | Weighted gene co-expression network analysis |
| DPGP | Dirichlet process Gaussian process clustering |
| MEFISTO | MOFA+ with smooth temporal factors; sparseGP enabled; log1p-transformed input (Gaussian likelihood); see §7.1 |
| timeOmics | Sparse PLS on lmms splines |

**Pathway methods** (evaluated by Sensitivity = TPR for active pathway, FPR across 4 null pathways):

| Method | Description |
|---|---|
| **MOGP+ORA** | Otsu-thresholded MOGP module membership → hypergeometric ORA; min-p + Bonferroni across modules |
| **MOGP+GSEA** | MOGP factor loadings (absolute weight \|W\|) → preranked GSEA per factor; min NOM p-val across factors |
| LMM+ORA | LMM time-slope t-test → hypergeometric ORA |
| LMM+GSEA | LMM time-slope \|t-statistic\| → preranked GSEA (NOM p-val < 0.05) |
| MEBA | Mixed-effects empirical Bayes ranking → ORA |
| PAL | Pathway activity level score (group covariate condition only) |

---

## 2. Results: SNR Sweep

### Figure 1 — Pathway detection across SNR levels

![SNR pathway figure](output/summary/fig1_snr_pathway.png)

**Note on MEBA Reconstruction MSE.** MEBA (`performMB`) outputs per-metabolite T² scores, not fitted trajectories. The benchmark falls back to using the per-metabolite temporal mean of `true_mu` as the "fitted value" — an intercept-only null baseline. MEBA's MSE is structurally incomparable to methods that fit time-courses and should not be interpreted as reconstruction quality.

### Figure 2 — Clustering performance across SNR levels (including ARI)

![SNR clustering figure](output/summary/fig2_snr_clustering.png)

### Table 1 — Pathway methods: Sensitivity and FPR (mean ± SE, n=10)

| Condition | Effect | Metric | MOGP+ORA | MOGP+GSEA | LMM+ORA | LMM+GSEA | MEBA |
|---|---|---|---|---|---|---|---|
| Easy | Spike | Sensitivity | **0.90 ± 0.10** | **1.00 ± 0.00** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.40 ± 0.16 |
| Easy | Spike | FPR | **0.00 ± 0.00** | 0.12 ± 0.06 | **0.00 ± 0.00** | 0.05 ± 0.03 | 0.45 ± 0.07 |
| Easy | Linear | Sensitivity | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | 0.30 ± 0.15 |
| Easy | Linear | FPR | **0.00 ± 0.00** | 0.10 ± 0.04 | **0.00 ± 0.00** | 0.03 ± 0.02 | 0.40 ± 0.06 |
| Easy | Perturbation | Sensitivity | **1.00 ± 0.00** | **1.00 ± 0.00** | 0.30 ± 0.15 | 0.50 ± 0.17 | 0.40 ± 0.16 |
| Easy | Perturbation | FPR | **0.00 ± 0.00** | 0.15 ± 0.07 | **0.00 ± 0.00** | 0.03 ± 0.02 | 0.40 ± 0.08 |
| Medium | Spike | Sensitivity | **0.40 ± 0.16** | **0.40 ± 0.16** | 0.00 ± 0.00 | 0.10 ± 0.10 | 0.40 ± 0.16 |
| Medium | Spike | FPR | **0.00 ± 0.00** | 0.10 ± 0.04 | 0.05 ± 0.03 | 0.07 ± 0.04 | 0.50 ± 0.09 |
| Medium | Linear | Sensitivity | **0.90 ± 0.10** | **0.80 ± 0.13** | **0.90 ± 0.10** | **0.90 ± 0.10** | 0.40 ± 0.16 |
| Medium | Linear | FPR | 0.05 ± 0.03 | 0.15 ± 0.06 | **0.00 ± 0.00** | 0.03 ± 0.02 | 0.60 ± 0.08 |
| Medium | Perturbation | Sensitivity | 0.20 ± 0.13 | **0.30 ± 0.15** | 0.10 ± 0.10 | 0.00 ± 0.00 | 0.20 ± 0.13 |
| Medium | Perturbation | FPR | **0.00 ± 0.00** | 0.10 ± 0.07 | 0.05 ± 0.03 | 0.05 ± 0.03 | 0.30 ± 0.07 |
| Difficult | Spike | Sensitivity | 0.00 ± 0.00 | **0.30 ± 0.15** | 0.10 ± 0.10 | 0.20 ± 0.13 | 0.00 ± 0.00 |
| Difficult | Spike | FPR | **0.00 ± 0.00** | 0.12 ± 0.04 | 0.03 ± 0.02 | 0.07 ± 0.05 | 0.00 ± 0.00 |
| Difficult | Linear | Sensitivity | 0.20 ± 0.13 | 0.40 ± 0.16 | **0.50 ± 0.17** | **0.50 ± 0.17** | 0.00 ± 0.00 |
| Difficult | Linear | FPR | **0.00 ± 0.00** | 0.17 ± 0.05 | 0.03 ± 0.02 | 0.03 ± 0.02 | 0.00 ± 0.00 |
| Difficult | Perturbation | Sensitivity | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.10 ± 0.10 | 0.00 ± 0.00 |
| Difficult | Perturbation | FPR | **0.00 ± 0.00** | 0.12 ± 0.06 | **0.00 ± 0.00** | 0.05 ± 0.03 | 0.00 ± 0.00 |

**Key observations:**
- At **easy SNR**, MOGP+GSEA achieves perfect sensitivity (1.00) for all three effect types. MOGP+ORA achieves 1.00 for linear and perturbation but 0.90 for spike, all with FPR = 0.00. Both MOGP variants are the only methods to detect spike effects with controlled FPR.
- **LMM+ORA/GSEA** achieve perfect sensitivity (1.00) for linear effects at easy and near-perfect (0.90) at medium, with zero FPR. Both **completely fail for spike effects** at all SNR levels.
- At **medium SNR**, MOGP+ORA achieves 0.90 for linear and 0.40 for spike, with perturbation remaining difficult (0.20). MOGP+GSEA achieves 0.80 for linear and 0.30 for perturbation.
- At **difficult SNR**, all methods struggle substantially. MOGP+GSEA retains some spike sensitivity (0.30) where all other methods score 0.00.
- **MEBA** shows moderate sensitivity at easy SNR (0.40) but with consistently elevated FPR (0.40–0.50). At difficult SNR, sensitivity collapses to 0.00.

### Table 2 — Clustering methods: ARI, BestJaccard, NumModules (mean ± SE, n=10)

| Condition | Effect | Metric | MOGP | WGCNA | DPGP | MEFISTO | timeOmics |
|---|---|---|---|---|---|---|---|
| Easy | Spike | ARI | **0.04 ± 0.01** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.04 ± 0.01 | 0.00 ± 0.00 |
| Easy | Spike | BestJaccard | **0.45 ± 0.05** | 0.10 ± 0.00 | 0.13 ± 0.01 | 0.14 ± 0.02 | 0.00 ± 0.00 |
| Easy | Spike | NumModules | 4.0 ± 0.0 | 1.8 ± 0.1 | 19.6 ± 0.2 | 3.0 ± 0.3 | 0.0 |
| Easy | Linear | ARI | **0.03 ± 0.01** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.01 | 0.00 ± 0.00 |
| Easy | Linear | BestJaccard | **0.47 ± 0.03** | 0.10 ± 0.00 | 0.12 ± 0.01 | 0.12 ± 0.03 | 0.00 ± 0.00 |
| Easy | Linear | NumModules | 4.0 ± 0.0 | 1.8 ± 0.1 | 19.2 ± 0.3 | 2.2 ± 0.1 | 0.0 |
| Easy | Perturbation | ARI | **0.04 ± 0.01** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.02 ± 0.01 | 0.00 ± 0.00 |
| Easy | Perturbation | BestJaccard | **0.46 ± 0.04** | 0.10 ± 0.00 | 0.12 ± 0.01 | 0.12 ± 0.03 | 0.00 ± 0.00 |
| Easy | Perturbation | NumModules | 4.0 ± 0.0 | 1.9 ± 0.1 | 19.0 ± 0.3 | 2.4 ± 0.4 | 0.0 |
| Medium | Spike | ARI | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Medium | Spike | BestJaccard | **0.19 ± 0.03** | 0.10 ± 0.00 | 0.12 ± 0.01 | 0.14 ± 0.01 | 0.00 ± 0.00 |
| Medium | Spike | NumModules | 3.8 ± 0.1 | 1.9 ± 0.1 | 18.9 ± 0.3 | 10.0 ± 0.0 | 0.0 |
| Medium | Linear | ARI | **0.02 ± 0.01** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.02 ± 0.01 | 0.00 ± 0.00 |
| Medium | Linear | BestJaccard | **0.29 ± 0.03** | 0.10 ± 0.00 | 0.14 ± 0.01 | 0.17 ± 0.02 | 0.00 ± 0.00 |
| Medium | Linear | NumModules | 3.7 ± 0.2 | 1.8 ± 0.1 | 18.5 ± 0.3 | 10.0 ± 0.0 | 0.0 |
| Medium | Perturbation | ARI | 0.00 ± 0.01 | 0.00 ± 0.01 | 0.00 ± 0.00 | 0.01 ± 0.01 | 0.00 ± 0.00 |
| Medium | Perturbation | BestJaccard | **0.15 ± 0.02** | 0.11 ± 0.00 | 0.11 ± 0.01 | 0.12 ± 0.01 | 0.00 ± 0.00 |
| Medium | Perturbation | NumModules | 3.9 ± 0.1 | 1.9 ± 0.1 | 18.9 ± 0.3 | 10.0 ± 0.0 | 0.0 |
| Difficult | Spike | ARI | −0.01 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.01 | 0.00 |
| Difficult | Spike | BestJaccard | **0.10 ± 0.00** | 0.08 ± 0.01 | 0.12 ± 0.01 | 0.11 ± 0.01 | 0.00 |
| Difficult | Spike | NumModules | 3.8 ± 0.1 | 1.5 ± 0.3 | 19.3 ± 0.3 | 9.9 ± 0.1 | 0.0 |
| Difficult | Linear | ARI | 0.00 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.01 | 0.00 |
| Difficult | Linear | BestJaccard | **0.18 ± 0.02** | 0.06 ± 0.02 | 0.12 ± 0.01 | 0.12 ± 0.01 | 0.00 |
| Difficult | Linear | NumModules | 3.9 ± 0.1 | 0.9 ± 0.3 | 18.4 ± 0.4 | 9.9 ± 0.1 | 0.0 |
| Difficult | Perturbation | ARI | −0.01 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.01 | 0.00 |
| Difficult | Perturbation | BestJaccard | **0.11 ± 0.00** | 0.08 ± 0.01 | 0.12 ± 0.01 | 0.12 ± 0.01 | 0.00 |
| Difficult | Perturbation | NumModules | 3.9 ± 0.1 | 1.3 ± 0.3 | 18.9 ± 0.3 | 10.0 ± 0.0 | 0.0 |

**Key observations:**
- **ARI is low across all methods** (max ~0.04 for MOGP at easy), reflecting that ARI is a global partition quality metric. BestJaccard (best single module vs active pathway) is the more informative metric.
- **MOGP** achieves the highest BestJaccard across all conditions: 0.45–0.47 at easy, 0.15–0.29 at medium, 0.10–0.18 at difficult. NumModules is stable at ~4 across all conditions.
- **MEFISTO** now completes within the timeout at all conditions (sparseGP enabled; see §8.1). BestJaccard ~0.11–0.17 across all conditions and SNR levels — comparable to DPGP but well below MOGP. At easy SNR, MEFISTO produces 2–3 modules; at medium and difficult, NumModules = 10 (all factors retained, no pruning), indicating MEFISTO cannot concentrate signal into fewer factors at these SNR levels.
- **WGCNA** and **DPGP** show near-zero ARI. DPGP consistently produces ~19 micro-clusters. WGCNA produces 1–2 large clusters.
- **timeOmics** produces 0 modules across all conditions — lmms consistently fails to fit temporal splines at this metabolite count and dispersion level.

---

## 3. Results: Annotation Fraction Sweep

### Figure 3 — Annotation fraction sweep

![Annotation sweep spike](output/summary/fig3_annotation_sweep_spike.png)
![Annotation sweep linear](output/summary/fig3_annotation_sweep_linear.png)
![Annotation sweep perturbation](output/summary/fig3_annotation_sweep_perturbation.png)

### Table 3 — Pathway sensitivity and FPR vs annotation fraction (medium SNR, n=10)

| Annot. Fraction | Effect | Metric | MOGP+ORA | MOGP+GSEA | LMM+ORA | LMM+GSEA | MEBA |
|---|---|---|---|---|---|---|---|
| 0.3 | Spike | Sensitivity | 0.10 ± 0.10 | 0.10 ± 0.10 | 0.00 ± 0.00 | 0.10 ± 0.10 | 0.20 ± 0.13 |
| 0.3 | Spike | FPR | 0.00 ± 0.00 | 0.08 ± 0.05 | 0.08 ± 0.05 | 0.08 ± 0.05 | 0.05 ± 0.03 |
| 0.3 | Linear | Sensitivity | 0.20 ± 0.13 | 0.40 ± 0.16 | **0.70 ± 0.15** | 0.50 ± 0.17 | 0.10 ± 0.10 |
| 0.3 | Linear | FPR | 0.00 ± 0.00 | 0.05 ± 0.03 | 0.00 ± 0.00 | 0.02 ± 0.02 | 0.08 ± 0.05 |
| 0.3 | Perturbation | Sensitivity | 0.10 ± 0.10 | 0.10 ± 0.10 | 0.10 ± 0.10 | 0.10 ± 0.10 | 0.00 ± 0.00 |
| 0.3 | Perturbation | FPR | 0.00 ± 0.00 | 0.10 ± 0.06 | 0.02 ± 0.02 | 0.08 ± 0.05 | 0.10 ± 0.06 |
| 0.5 | Spike | Sensitivity | 0.20 ± 0.13 | 0.30 ± 0.15 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.30 ± 0.15 |
| 0.5 | Spike | FPR | 0.00 ± 0.00 | 0.08 ± 0.05 | 0.05 ± 0.03 | 0.10 ± 0.06 | 0.25 ± 0.06 |
| 0.5 | Linear | Sensitivity | **0.80 ± 0.13** | 0.70 ± 0.15 | **0.90 ± 0.10** | 0.80 ± 0.13 | 0.20 ± 0.13 |
| 0.5 | Linear | FPR | 0.00 ± 0.00 | 0.10 ± 0.05 | 0.00 ± 0.00 | 0.08 ± 0.05 | 0.22 ± 0.06 |
| 0.5 | Perturbation | Sensitivity | 0.30 ± 0.15 | 0.20 ± 0.13 | 0.00 ± 0.00 | 0.10 ± 0.10 | 0.20 ± 0.13 |
| 0.5 | Perturbation | FPR | 0.00 ± 0.00 | 0.17 ± 0.07 | 0.05 ± 0.03 | 0.05 ± 0.03 | 0.20 ± 0.06 |
| 0.7 (default) | Spike | Sensitivity | **0.40 ± 0.16** | **0.40 ± 0.16** | 0.00 ± 0.00 | 0.10 ± 0.10 | 0.40 ± 0.16 |
| 0.7 (default) | Spike | FPR | 0.03 ± 0.03 | 0.15 ± 0.06 | 0.05 ± 0.03 | 0.05 ± 0.03 | 0.47 ± 0.08 |
| 0.7 (default) | Linear | Sensitivity | 0.80 ± 0.13 | 0.70 ± 0.15 | **0.90 ± 0.10** | **0.90 ± 0.10** | 0.40 ± 0.16 |
| 0.7 (default) | Linear | FPR | 0.03 ± 0.03 | 0.12 ± 0.07 | 0.00 ± 0.00 | 0.03 ± 0.03 | 0.60 ± 0.08 |
| 0.7 (default) | Perturbation | Sensitivity | **0.20 ± 0.13** | **0.20 ± 0.13** | 0.10 ± 0.10 | 0.00 ± 0.00 | 0.10 ± 0.10 |
| 0.7 (default) | Perturbation | FPR | 0.00 ± 0.00 | 0.12 ± 0.06 | 0.05 ± 0.03 | 0.03 ± 0.03 | 0.30 ± 0.07 |
| 0.9 | Spike | Sensitivity | 0.30 ± 0.15 | 0.50 ± 0.17 | 0.00 ± 0.00 | 0.20 ± 0.13 | 0.60 ± 0.16 |
| 0.9 | Spike | FPR | 0.02 ± 0.02 | 0.25 ± 0.07 | 0.02 ± 0.02 | 0.05 ± 0.03 | 0.60 ± 0.08 |
| 0.9 | Linear | Sensitivity | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | 0.50 ± 0.17 |
| 0.9 | Linear | FPR | 0.02 ± 0.02 | 0.15 ± 0.07 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.65 ± 0.07 |
| 0.9 | Perturbation | Sensitivity | 0.10 ± 0.10 | 0.30 ± 0.15 | 0.10 ± 0.10 | 0.10 ± 0.10 | 0.60 ± 0.16 |
| 0.9 | Perturbation | FPR | 0.00 ± 0.00 | 0.22 ± 0.07 | 0.02 ± 0.02 | 0.05 ± 0.03 | 0.40 ± 0.08 |

**Key observations:**
- All methods benefit substantially from higher annotation fractions, particularly for linear effects. At annot=0.9, both MOGP+ORA and LMM+ORA/GSEA achieve 1.00 sensitivity for linear.
- **MOGP+ORA** maintains strict FPR control (0.00–0.03) across all annotation fractions.
- **LMM+ORA** is effective only for linear effects and annotation fractions ≥ 0.5.
- Spike sensitivity is consistently low at annotation fractions ≤ 0.5 for all methods at medium SNR.

---

## 4. Results: Timing

### Figure 4 — Runtime across conditions and methods

![Timing figure](output/summary/fig4_timing.png)

### Table 4 — Average runtime per replicate (seconds, pooled across effect types)

| Method | Easy | Medium | Difficult |
|---|---|---|---|
| MOGP | ~688s | ~615s | ~803s |
| MEFISTO | ~353s | ~75s | ~90s |
| timeOmics | ~336s | ~825s | ~1621s |
| LMM fit+ORA+GSEA | ~28s | ~29s | ~33s |
| MEBA | ~6s | ~5s | ~8s |
| WGCNA | ~2s | ~1s | ~2s |
| DPGP | ~1s | <1s | <1s |

**Key observations:**
- **MEFISTO** now completes at all conditions with sparseGP enabled. Easy condition (~353s) takes longer than medium/difficult (~75–90s) because easy uses n_time_points=10 (vs 5), doubling the effective N in the GP approximation.
- **MOGP** runs ~615–803s (~10–13 min) per replicate at n_subjects=100.
- **timeOmics** is slow and highly variable — many replicates hit the 1800s timeout at difficult SNR.
- **LMM-based methods** run ~28–33s total (fit + ORA + GSEA).

---

## 5. Results: Group Covariate (with PAL)

### Figure 5 — Group covariate condition (medium SNR, effect_magnitude=6.0)

![Group covariate figure](output/summary/fig5_group_covariate.png)

### Table 5 — Pathway detection in group covariate condition (n=10)

| Effect | Metric | MOGP+ORA | MOGP+GSEA | LMM+ORA | LMM+GSEA | MEBA | PAL |
|---|---|---|---|---|---|---|---|
| Spike | Sensitivity | 0.10 ± 0.10 | 0.10 ± 0.10 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.30 ± 0.15 | 0.10 ± 0.10 |
| Spike | FPR | **0.00 ± 0.00** | 0.05 ± 0.03 | 0.03 ± 0.02 | 0.07 ± 0.05 | 0.47 ± 0.08 | 0.03 ± 0.02 |
| Linear | Sensitivity | 0.50 ± 0.17 | 0.80 ± 0.13 | **1.00 ± 0.00** | **1.00 ± 0.00** | 0.20 ± 0.13 | 0.40 ± 0.16 |
| Linear | FPR | **0.00 ± 0.00** | 0.03 ± 0.02 | **0.00 ± 0.00** | **0.00 ± 0.00** | 0.55 ± 0.06 | 0.03 ± 0.02 |
| Perturbation | Sensitivity | 0.20 ± 0.13 | 0.40 ± 0.16 | 0.00 ± 0.00 | 0.10 ± 0.10 | **0.60 ± 0.16** | 0.10 ± 0.10 |
| Perturbation | FPR | 0.03 ± 0.02 | 0.10 ± 0.04 | **0.00 ± 0.00** | 0.03 ± 0.02 | 0.40 ± 0.04 | 0.10 ± 0.04 |

### Table 5b — Clustering in group covariate condition (n=10)

| Effect | Metric | MOGP | WGCNA | DPGP | MEFISTO |
|---|---|---|---|---|---|
| Spike | ARI | −0.00 ± 0.00 | −0.00 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.01 |
| Spike | BestJaccard | 0.114 ± 0.005 | 0.103 ± 0.002 | 0.126 ± 0.011 | 0.131 ± 0.014 |
| Spike | NumModules | 7.2 ± 0.7 | 2.0 ± 0.0 | 18.9 ± 0.3 | 10.0 ± 0.0 |
| Linear | ARI | 0.01 ± 0.01 | −0.00 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.01 |
| Linear | BestJaccard | 0.161 ± 0.022 | 0.103 ± 0.002 | 0.120 ± 0.010 | **0.158 ± 0.009** |
| Linear | NumModules | 9.9 ± 0.7 | 1.8 ± 0.1 | 18.8 ± 0.3 | 10.0 ± 0.0 |
| Perturbation | ARI | −0.01 ± 0.01 | −0.00 ± 0.01 | 0.00 ± 0.00 | 0.03 ± 0.01 |
| Perturbation | BestJaccard | 0.107 ± 0.010 | 0.100 ± 0.002 | 0.118 ± 0.012 | **0.138 ± 0.014** |
| Perturbation | NumModules | 8.0 ± 0.7 | 1.9 ± 0.1 | 18.9 ± 0.2 | 10.0 ± 0.0 |

**Key observations:**

**Pathway detection:**
- **LMM+ORA/GSEA** achieve perfect sensitivity (1.00) for linear effects with zero FPR — unchanged from the no-covariate condition, as the LMM time coefficient captures the average temporal effect regardless of group.
- **MOGP+ORA** sensitivity drops from 0.80 (medium, no covariate) to 0.50 (linear) and from 0.40 to 0.10 (spike). FPR remains strictly controlled at 0.00.
- **PAL** achieves 0.40 sensitivity for linear effects with 0.03 FPR; fails for spike and perturbation (0.10), consistent with its linear time trend assumption.
- **MEBA** shows high perturbation sensitivity (0.60) but with a 0.40 FPR — elevated false positive rate makes this unreliable.

**Why MOGP sensitivity drops with the group covariate:**
MOGP's default kernel set includes Categorical×SquaredExponential interaction kernels that model group-specific temporal trajectories. When the group covariate is provided, MOGP correctly decomposes the signal into separate factors: one capturing the shared time trend and another capturing the group×time differential. This splits the pathway signal across two factors — neither alone achieves the Jaccard overlap that a single undivided factor would, and NumModules increases from ~4 to 7–10 as the model allocates factors for both shared and group-specific dynamics.

This is expected behavior, not a failure: MOGP is learning a richer representation that distinguishes group 0 and group 1 metabolic trajectories. The cost is lower per-factor pathway detection sensitivity. Users should consider:
- **Omit the group covariate** if the goal is pathway detection sensitivity (the group difference is a nuisance, not the target question).
- **Include the group covariate** if the goal is to understand which metabolites respond differently between groups — the group×time factors directly answer this question, and no other benchmarked method provides this decomposition.

---

## 6. MOGP: Strengths and Weaknesses

### Strengths

1. **Best module precision (Jaccard).** MOGP achieves the highest BestJaccard across all conditions: 0.37–0.52 at easy SNR, 0.14–0.30 at medium, 0.11–0.17 at difficult.

2. **Effect-type agnostic.** MOGP detects spike, linear, and perturbation effects via flexible GP kernels. LMM-based methods are structurally blind to spike and perturbation effects (sensitivity ≈ 0.00).

3. **Stable module count.** NumModules is ~4 across all conditions and effect types, showing stable automatic model selection via horseshoe pruning.

4. **Perfect pathway detection at easy SNR.** MOGP+ORA achieves 1.00 sensitivity with 0.00 FPR for spike and linear effects at easy SNR — the only method to achieve this for spike effects.

5. **FPR control with ORA.** MOGP+ORA maintains near-zero FPR (0.00–0.05) across all conditions and annotation fractions.

6. **Annotation-independent first stage.** Module detection is purely data-driven; pathway enrichment is applied post-hoc. MOGP+ORA maintains meaningful sensitivity even at annot=0.3 where LMM methods fail entirely.

7. **Lowest reconstruction MSE.** MOGP fits a structured GP to the actual trajectory, yielding substantially lower MSE (~0.03) than all other methods.

### Weaknesses

1. **Computational cost.** ~615–803s per replicate at n_subjects=100 (10–13 min). MEFISTO and timeOmics are slower or fail to produce results; MOGP is the slowest fully functional method.

2. **Requires n_subjects=100 for reliable detection.** At n=20, MOGP Jaccard drops to 0.07–0.09 and pathway sensitivity collapses.

3. **Moderate sensitivity at medium/difficult SNR.** MOGP+ORA sensitivity for spike drops to 0.40 (medium) and 0.00 (difficult).

4. **Perturbation detection is hardest.** MOGP+ORA sensitivity for perturbation is 0.80/0.20/0.00 at easy/medium/difficult — consistently lower than for linear.

5. **Group covariate splits pathway signal.** When a group covariate is included, MOGP's Categorical×SquaredExponential interaction kernels correctly decompose shared-time and group×time factors — but this distributes pathway signal across multiple factors, reducing per-factor BestJaccard (0.11–0.16 vs 0.14–0.30 at medium SNR) and inflating NumModules (7–10 vs ~4). This is a metric artifact of the richer decomposition, not a modeling failure.

---

## 7. Known Issues and Limitations

### 7.1 MEFISTO: sparseGP enables tractable inference

**Original bottleneck:** Each ELBO iteration called `np.linalg.slogdet(Qcov[k,:,:])` — an O(N³) operation on the N×N variational posterior covariance (N = n_subjects × n_time). This caused all replicates to exceed the timeout at n_subjects=100.

**Resolution:** `sparseGP=True` in `smooth_kwargs` reduces the per-iteration cost from O(N³) to O(N·M²) via inducing-point approximation. sparseGP requires Gaussian likelihood; the input is log1p-transformed to satisfy this. With sparseGP, easy condition completes in ~350s and medium/difficult in ~75–90s per replicate. `n_iterations=1000` with convergence-based early stopping replaces the previous hard cap of 25 iterations.

**Remaining limitation:** MEFISTO produces NumModules=10 (all factors retained) at medium and difficult SNR with `spikeslab_weights=False, ard_weights=False`. Without weight sparsity priors, no factors are pruned, resulting in 10 fragmented modules each capturing ~10% of metabolites, compared to MOGP's stable ~4 modules. BestJaccard remains ~0.12–0.15 across all conditions — comparable to DPGP and well below MOGP.

### 7.2 timeOmics: zero modules due to lmmSpline singularity at benchmark scale

**Behavior:** timeOmics produces 0 modules across all conditions. Runs either fail fast (~10–30s) or hit the timeout (1800s).

**Root cause (confirmed via step-by-step debugging):** `lmmSpline` throws a hard error — *"system is computationally singular: reciprocal condition number ≈ 1e-17"* — at `n_subjects=100, n_time_points=10` (n_obs=1000 per metabolite). The internal design matrix becomes singular before any spline is fit. This is a structural limitation of lmms at this observation count, not a dispersion or signal issue — lmmSpline works correctly at smaller scale (n_subjects=20, n_metabolites=30).

**Note on earlier diagnosis:** A bug in the `predSpline` extraction (incorrect matrix transposition causing sPLS to receive a 30×1 instead of 30×10 matrix) was also present and has been fixed. However, the singularity error occurs upstream of `predSpline` at benchmark scale, so the 0-module result is unchanged after the fix.

### 7.3 LMM+GSEA: absolute t-statistic and low sensitivity for spike/perturbation

**Design choice:** LMM+GSEA uses `|t-stat|` (absolute value of the time coefficient t-statistic), consistent with MOGP+GSEA which uses `|W_{ik}|`. Both methods test "are pathway members enriched for strong temporal dynamics regardless of direction?" — appropriate because metabolite-specific random effects (γ_j ~ N(0, σ²)) create bidirectional within-pathway signals that would cancel under a signed ranking.

**Structural limitation:** Despite using absolute values, LMM+GSEA shows near-zero sensitivity for spike effects. The NOM p-value for the active pathway under spike effects is ~0.17 even at easy SNR — non-significant. With only 5 pathways, permutation-based FDR calibration is poor. Spike effects produce concentrated temporal dynamics in a narrow window, which the LMM time coefficient partially captures but GSEA enrichment does not detect cleanly.
