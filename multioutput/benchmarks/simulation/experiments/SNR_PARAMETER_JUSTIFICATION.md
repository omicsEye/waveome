# SNR Parameter Justification for Simulation Benchmark

**Date:** 2026-03-30

---

## 1. Background

The simulation benchmark evaluates longitudinal metabolomics methods across three SNR conditions
(easy, medium, difficult). This document records the rationale for the chosen parameter values,
calibrated against the iHMP metabolomics dataset and relevant literature.

---

## 2. Dispersion Calibration (iHMP)

NB dispersion parameters were calibrated against the iHMP metabolomics data
(`iHMP_metabolomics.csv`, 81,867 metabolite features × 546 samples).

We fit a Negative Binomial model to each metabolite via method-of-moments:

```
r = μ² / (var - μ)     [variance = μ + μ²/r]
```

Summary of fitted r values across metabolites:

| Percentile | r (dispersion) | α = 1/r |
|---|---|---|
| 5th | 0.025 | 40.3 |
| 25th | 0.105 | 9.5 |
| 50th (median) | 0.254 | 3.9 |
| 75th | 0.537 | 1.9 |
| 90th | 1.033 | 1.0 |
| 95th | 1.521 | 0.7 |

A LogNormal fit to r gives:
- **Median r = 0.23** (= `dispersion` parameter)
- **σ(log r) = 1.24** (= `dispersion_spread` parameter)

These values are fixed across all three SNR conditions — dispersion is a property of the
measurement platform, not the biological condition being studied.

Note: the α = 1/r parameterization from the iHMP notebook output reported a range of
[0.23, 230.9], consistent with our empirical estimates (median α ≈ 3.9, i.e. r ≈ 0.25).

---

## 3. SNR Definition

For a single active-pathway metabolite, the latent signal on the log scale is:

```
log(μ_{it}) = subject_intercept_i + pathway_intercept_i + metabolite_baseline
              + effect(t) × metabolite_scaler
```

where `metabolite_scaler ~ N(0, metabolite_effect_sd²)` with `metabolite_effect_sd = 0.2`.

The **signal amplitude** (averaged over the scaler distribution) is:

```
signal ≈ effect_magnitude × metabolite_effect_sd
```

The **noise** is dominated by NB observation variance. At iHMP dispersion levels (r ≈ 0.23),
the log-scale CV is large (~0.6 log-units), completely dominating the pathway random effect
(SD = 0.3). The effective SNR is:

```
SNR = mean(|signal|) / mean(std(log(obs+1)))
```

computed empirically via Monte Carlo over 2,000–3,000 metabolite draws.

Subject random effects (`subject_noise`) contribute to the noise denominator when large
enough to exceed the NB floor. We use this as the primary difficulty lever alongside
`effect_magnitude`, fixing `subject_noise = 0.5` across all conditions as biologically
plausible (consistent with Piening et al. 2018, who observed subject-level metabolite
shifts of 0.3–1.5 log2 units ≈ 0.2–1.0 log-units over physiological perturbations).

---

## 4. Chosen Parameters

| Condition | `effect_magnitude` | `subject_noise` | `dispersion` | `dispersion_spread` | Empirical SNR |
|---|---|---|---|---|---|
| Easy | 8 | 0.5 | 0.23 | 1.24 | ~1.9 |
| Medium | 4 | 0.5 | 0.23 | 1.24 | ~1.0 |
| Difficult | 2 | 0.5 | 0.23 | 1.24 | ~0.5 |

These SNR values are consistent with the range considered "challenging" in related
benchmarks (DPGP: McDowell et al. 2018; MEFISTO: Argelaguet et al. 2021), where
SNR 1–3 on the linear scale represents typical experimental conditions.

### Why effect_magnitude drives SNR rather than subject_noise

At iHMP dispersion levels, the NB noise floor (~0.6 log-units) dominates subject-level
variance unless `subject_noise` is made unrealistically large (>1.0, implying ±4 log-unit
subject ranges, i.e. >50x fold range in baseline expression). Varying `effect_magnitude`
alone produces a clean, interpretable SNR gradient without introducing biologically
implausible subject heterogeneity.

---

## 5. Derived Experiments

Two additional experiments use medium SNR parameters as their baseline.

### Annotation quality sweep (`run_annotation.bash`)

Runs at medium SNR (`effect_magnitude=4.0`, `subject_noise=0.5`) while varying
`annotation_fraction` across {0.3, 0.5, 0.9}. The goal is to isolate the effect of
incomplete pathway annotation on detection performance, holding SNR constant.

### Group covariate experiment (`run_group_covariate.bash`)

Adds a binary group variable where group 1 receives 50% of the temporal effect.
`effect_magnitude=6.0` is chosen so that the average effective magnitude across groups
is `0.75 × 6.0 = 4.5 ≈ 4.0`, keeping the population-average SNR near medium (~1.0).
Group 0 sees SNR ~1.5 and group 1 sees SNR ~0.75.

---

## 6. Literature References

- **Piening et al. (2018)** *Cell* — longitudinal multi-omics profiling; observed
  physiological metabolite shifts of 0.3–1.5 log2 units, motivating our subject noise
  range (0.2–1.0 log-units).

- **McDowell et al. (2018)** *PLOS Computational Biology* — DPGP benchmark; "easy"
  defined as noise SD ≈ 0.1 of signal amplitude; SNR 1–3 considered challenging.

- **Argelaguet et al. (2021)** *Genome Biology* — MEFISTO; variance explained ≥5% used
  as the proxy for detectable factors, consistent with our SNR ~0.5–2 range.

- **Soneson & Delorenzi (2013)** *BMC Bioinformatics* — log2 FC = 1.0 (2x fold change)
  as conventional "detectable" threshold; borrowed by metabolomics simulation benchmarks.

- **iHMP Research Consortium (2019)** *Nature* — source of iHMP metabolomics data used
  for dispersion calibration.
