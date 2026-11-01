# Harder simulation regimes: design for review

Status: **draft for author sign-off**. Nothing here is implemented yet. Section 9
lists the decisions that need an answer before any generator code is written.

Covers the reviewer items that change the simulation:

| Item | Ask | Covered by |
|---|---|---|
| R1.M2 / E1 | Harder settings; stronger baselines | Regimes A–E (sections 3–7); GPcounts and LonGP (section 8) |
| R2.m4 | Correlated covariates: selection and deviance split | Regime C (one design answers both) |
| R1.M3, R2.4 | Subject-wise and forward-in-time splits; unseen subjects | Split runs (section 7) |
| Tracker 8 / R1.M4 | NB vs log-normal vs ZINB by holdout KL | Regime D |
| R2.m1 | Number of candidate components and true non-nulls | Truth table recorded in every regime cell |
| R1.m6 | Fit fallback and failure rates | Status column recorded in every regime cell |

## 1. What stays fixed

Frozen decisions carry over unchanged:

- refit-ΔBIC statistic;
- permutation null;
- BH at q ≤ 0.10, pooled over the replicates of each setting;
- `id` not tested;
- interactions tested under each covariate's permutation;
- two-level scoring;
- 50 replicates per setting.

The main grid (Figs 3–4) is untouched. Each regime is an extra set of settings with its
own outputs.

**Code location.** A new script, `examples/simulations/sim_regimes_hpc_run.py`, with its
own sbatch wrapper.

- It imports the shared pieces (`StageResources`, `provenance`, `calc_kl_all`, the true
  kernels) from `sim_waveome_hpc_run.py` without editing that file.
- Each array task of the main sweep reads `sim_waveome_hpc_run.py` when it starts, so the
  file stays exactly as the sweep began.
- The cost is some duplication. Every baseline in `run_simulation` hard-codes the formula
  `time + treat`: mixed model, NB-GLM, GAM, lasso, and the ARD design matrix. Regimes C
  and E need those baselines written for a list of covariates. The regime script carries
  that generalized copy.
- No `waveome/` library change is planned except possibly for regime D (section 6). Any
  library change waits until the main sweep has finished, because the shared venv is
  frozen while the sweep runs.

**Pairing with the main grid.** Regimes C, D and E start from the main data-generating
process (DGP), using the same seed as the matching main-grid cell (`random_seed = run_id`).
Everything they add comes from a separate generator, `np.random.default_rng([9102, run_id])`,
drawn after the main data.

- The underlying data are therefore identical to the main-grid cell, and the regime's
  effect is a paired difference.
- With severity set to zero, the generator must reproduce `sim_data` exactly. This is an
  acceptance check.

## 2. Settings: sizes and noise

Three main-grid settings, so the main sweep already supplies each regime's reference
condition. No extra reference runs are needed.

| Training obs | units | rate | Why |
|---|---|---|---|
| 160 | 50 | 4 | small, typical of thin omics cohorts |
| 640 | 100 | 8 | middle |
| 1,600 | 500 | 4 | largest affordable; matches the measured pilot cell |

The 6,400 cells are left out: they are about 60% of the main sweep's cost.

**Noise.** Fixed at `epsilon = 1`, `alpha = 1`. The regime should be the only source of
difficulty, so it runs where the main-grid methods have signal to lose. This is provisional:
if the main sweep shows waveome already struggling at that setting, the anchor moves.
**(Decision 1.)**

**Signal strength.**

- For deterministic shapes (regimes A and B), every non-`id` effect is scaled so its
  variance on the latent log scale is 1.0, measured over a dense grid of raw time [0, 12].
  That is roughly the size of the main-grid components (kernel variances 0.5–2), so
  "harder" means harder shape, not weaker signal.
- The `id` intercept has variance 1.0.
- Effects are defined on raw time [0, 12] and then time is standardized, as `sim_data`
  does.

## 3. Regime A: misspecified shapes (R1.M2)

**Stresses:** the true function is not a draw from any kernel waveome can propose.

**Vocabulary.**

- The penalized path uses only SE components (with categorical interactions).
- The search path uses SE, Matern12, Linear and Periodic.
- The main grid already has some mismatch: the y1, y2 and y4 truths are Matern12, Periodic
  and Polynomial, against an SE-only penalized vocabulary. But every main-grid outcome is a
  stationary GP draw.

**Design.** This regime uses deterministic, nonstationary shapes ported from
`multioutput/benchmarks/simulation/effects.py`. The functions are copied into the regime
script; `multioutput/` stays read-only.

| Outcome | Latent mean (raw time) | Covariate truth |
|---|---|---|
| y1 | id + pulse on [5, 6.5) (`create_spike_effect`) | time |
| y2 | id + ramp 3→7 then plateau (`create_trapezoid_effect`) | time |
| y3 | id + treat × jump at 4 with half-life 1.5 decay (`create_perturbation_effect`), treated units only | treat, time |
| y4 | id only | none (fully null outcome) |

- **Severity:** one level. The difficulty is the shape, not a dial.
- **Scoring:** covariate level only. No true kernel exists, so term-level scoring is not
  defined.

## 4. Regime B: change points (R1.M2)

**Stresses:** abrupt, persistent level shifts that smooth stationary kernels cannot
represent.

| Outcome | Latent mean | Covariate truth |
|---|---|---|
| y1 | id + step at t = 6 (population) | time |
| y2 | id + treat × step at t = 6 | treat, time |
| y3 | id + step at a unit-specific onset τᵢ ~ U(3, 9) | time (the onset varies by unit, i.e. id × time; `id` is not tested) |
| y4 | id + constant treat shift | treat (time null) |

- y3 is the flare-onset pattern from the application.
- y4 checks that a constant group difference is not reported as a time effect.
- **Severity:** one level.
- **Scoring:** covariate level only.

## 5. Regime C: correlated null covariate (R1.M2 nuisance/confounding + R2.m4)

**Stresses:** false-positive control and attribution when an irrelevant covariate tracks a
real one.

**Design.**

- Main DGP (k1–k4, unchanged), plus a time-varying covariate
  `z = ρ·time + √(1−ρ²)·e`, with e ~ N(0, 1) per observation.
- `z` has no effect on any outcome; time is true in all four.
- This mirrors the iHMP pair of days-from-max-HBI and time-in-study, which are correlated by
  construction (R1.m7).
- **Severity:** ρ ∈ {0.3, 0.6, 0.9}.
- **Tested covariates:** treat, time, z.

**Reported, per outcome and ρ:**

1. **Selection** (R2.m4 part 1). Across replicates, how often each outcome is called:
   - time only (right);
   - z only (wrong);
   - both;
   - neither.

   `z` calls feed the false-discovery rate.
2. **Deviance split** (R2.m4 part 2). Each cell saves
   `GPSearch.get_significance_table(include_significance=False)`, so the marginal
   `deviance_explained` of the time and z components is available. The main harness keeps
   only the permutation draws, which is not enough for this.

**Baselines.** `z` is added as a fixed effect in the mixed model, NB-GLM and lasso, as a
linear term in the GAM, and as an input dimension in ARD. Random effects stay as in the
main grid (id, id × time, id × treat), so the mixed-model random-slope search does not grow.

**Optional second pair (Decision 2):** a unit-level continuous covariate correlated with
`treat`, testing a binary/continuous pair. It adds about 50% to this regime's cost.

## 6. Regime D: zero inflation (R1.M2 + tracker item 8 / R1.M4)

**Stresses:** the likelihood, with zeros beyond what the negative binomial (NB) predicts.

**Design.**

- Main DGP, then each observation is set to 0 with probability π, independent of the
  covariates. Zeros are then noise, not signal.
- **Severity:** π ∈ {0, 0.25, 0.5}. π = 0 is the anchor for the likelihood comparison.
- **waveome fits:** three likelihoods, each with its own penalized fit, permutation null
  and pooled BH (the "≈3 calibration passes" in tracker item 8):
  - NB (as in the main grid);
  - zero-inflated NB (ZINB);
  - log-normal (Gaussian on `log1p`).
- **Scoring:** selection as in the main grid, plus holdout KL of each fitted predictive
  distribution against the true ZINB distribution.
- **KL for the ZINB truth.** `calc_kl_all` currently assumes the truth is NB with parameters
  (p, n). It needs a ZINB-truth variant. That change lives in the regime script.
- **Library check (T5).** `ZeroInflatedNegativeBinomial` exists (`waveome/likelihoods.py`,
  selectable as `"zeroinflated_negativebinomial"` in `utilities.py`), but nobody has
  checked that it works end to end through the penalized fit, the permutation and the
  search. If it needs a library fix, regime D waits for the main sweep to finish (frozen
  venv).

**Optional (Decision 3):** a covariate-dependent π, with zeros more common in the treated
group. This is a harder confounding case, since an NB fit would credit the extra zeros to
`treat`.

## 7. Regime E: noise covariates (R1.M2) and split runs (R1.M3, R2.4)

### Regime E: higher-dimensional noise

**Stresses:** specificity as irrelevant inputs grow.

**Design.**

- Main DGP, plus p null covariates, cycling through three types:
  - unit-level binary, Bernoulli(0.5), entered as categorical like `treat`;
  - unit-level N(0, 1);
  - within-unit AR(1), φ = 0.8, marginal N(0, 1).
- The AR(1) ones are smooth within a unit, so they can mimic a trend. That is harder and
  more realistic than i.i.d. noise.
- **Severity:** p ∈ {2, 5, 10}.
- **Baselines:** as in regime C. With 13 covariates, the lasso's pairwise interactions are
  78 columns, which is fine.

**Cost grows with p** (Decision 4):

- every tested covariate gets its own permutation stream;
- each search depth tries every kernel × covariate.

At p = 10 the permutation stage is roughly 6× the main grid's (12 tested covariates
against 2). The proposal is 160 and 640 only for this regime.

### Split runs (R1.M3, R2.4)

**Design.**

- Main DGP at the three sizes, with two alternative 80/20 holdouts:
  - **subject-wise:** 20% of units held out entirely;
  - **forward-in-time:** observations above the 80th percentile of sampled time held out.
- The question is prediction: holdout KL and MSE per method.
- The subject-wise run doubles as the R2.4 illustration. For an unseen `id`, the
  categorical kernel contributes nothing, so the prediction is the population-level one.

**Baselines need unseen-id handling.**

- `get_mixed_preds` already sets a missing unit's random effect to zero.
- NB-GLM (`C(id, levels=id_list)`), GAM and ARD use `id` as fixed effects or dummies. An
  unseen unit must get the population-level prediction (zero `id` effect) instead.

**Permutation in split runs (Decision 5).** The split runs report prediction only, so the
proposal is to skip the permutation stage there and keep the search. This saves most of
the cost, but it departs from "same permutation strategy in every cell", so it needs your
call. Applying that rule to the selection-scored cells only would keep it intact.

## 8. Baselines: GPcounts and LonGP

### GPcounts

- **Install.** Adds RobustGP (markvdw, from GitHub), jug and json-tricks. Checked earlier:
  no pinned package changes, but it is untested on gpflow 2.9.1 / TF 2.15. Note that PyPI
  `robustgp` is a different package.
- **Mapping to covariate-level calls.** One-sample test (dynamic vs constant) gives the
  `time` call. Two-sample test (treated vs untreated) gives the `treat` call. Both are run
  under NB, and under ZINB in regime D.
- **Limitations, reported as such.** It does not model repeated measures (`id`) and cannot
  take extra covariates. In C and E it is scored on time and treat only. It cannot produce
  a false positive on `z` or the noise covariates, so it is not comparable on that axis.
- **Runs.** A separate pass over the main grid (data regenerated from the seed), and over
  every regime. Its own script, so the main sweep is unaffected.

### LonGP

- The closest competitor in concept: additive GP kernel selection with individual-specific
  components.
- It is MATLAB/GPstuff with MCMC; about 8 h per run is reported.
- **Proposal:** attempt it at 160 obs on a few replicates per regime. If integration is
  infeasible, document the attempt and the reason in the response.
- **Needs from you (Decision 6):** whether Pegasus has MATLAB with a license usable from
  batch jobs.

## 9. Grid, cost, and decisions

### Cell counts at 50 replicates

| Regime | Levels | Sizes | Cells | waveome fits per cell |
|---|---|---|---|---|
| A misspecified | 1 | 3 | 150 | 1 |
| B change point | 1 | 3 | 150 | 1 |
| C correlated | 3 (ρ) | 3 | 450 | 1 |
| D zero inflation | 3 (π) | 3 | 450 | 3 likelihoods |
| E noise covariates | 3 (p) | 2 | 300 | 1, with 2–6× permutation cost |
| Splits | 2 | 3 | 300 | 1 (no permutation, if Decision 5) |
| **Total** | | | **1,800** | |

### Cost: rough, to be replaced by measurement

Only two cells have been measured:

| Cell | Wall time | CPU-hours |
|---|---|---|
| 1,600-obs pilot | 17.5 h on 12 CPUs | ≈ 210 |
| 16-obs smoke cell | 497 s on 4 CPUs | ≈ 0.55 |

Scaling assumption, used only to size the request:

- the main stages scale with n · m², where m = min(n, 500) inducing points;
- so 640 obs ≈ 40% of 1,600;
- and 160 obs is dominated by fixed overhead, taken as 5–10 CPU-hours.

On that basis:

| Regime | CPU-hours |
|---|---|
| A, B | ≈ 15k each |
| C | ≈ 45k |
| D | ≈ 90k |
| E | ≈ 55k |
| Splits | ≈ 15–30k |
| **Total** | **≈ 250k**, about 20% of the main sweep's 1.2–1.5M |

GPcounts and LonGP are not included. Before the full submission, a measurement pilot (one
replicate per regime level at 640 and 1,600) will replace these numbers using each cell's
resources table.

### Decisions needed

1. **Noise anchor:** `epsilon = 1, alpha = 1` for every regime, revisited after the main
   sweep. Alternative: run each regime at two noise settings, which doubles the cost.
2. **Regime C second pair** (unit-level covariate correlated with `treat`): include or
   skip? Recommended: skip unless R2.m4 needs a binary example.
3. **Regime D covariate-dependent zeros:** include or skip? Recommended: skip. Constant π
   answers what the reviewers asked.
4. **Regime E sizes:** 160 and 640 only? Recommended: yes.
5. **Split runs without permutation:** accept that departure from "same permutation
   strategy in every cell"? It has to be your call, since it touches a frozen decision.
6. **LonGP:** is MATLAB available on Pegasus for batch jobs?
7. **Severity levels:** ρ ∈ {0.3, 0.6, 0.9}, π ∈ {0, 0.25, 0.5}, p ∈ {2, 5, 10}. Adjust
   if you have values matched to iHMP (for example, its observed zero fraction).

## 10. Build order (one task at a time, each with its acceptance check)

| # | Task | Acceptance check |
|---|---|---|
| R1 | Generators for A–E, with truth tables | Severity 0 reproduces `sim_data` exactly for the same seed; shape effects have latent variance 1.0; a plot of example draws per regime for your review |
| R2 | Regime runner: waveome plus covariate-list baselines; truth table, status column and significance table saved per cell | One smoke cell per regime runs locally; a main-DGP cell with no added covariates gives the same selections as `sim_waveome_hpc_run.py` |
| R3 | Split runs | Holdout sets are disjoint by unit (subject-wise) or by time (forward); unseen-id predictions equal the population-level ones |
| R4 | Regime D likelihoods and ZINB-truth KL (T5 check) | All three likelihoods fit, permute and produce selections on a smoke cell; KL is zero for the true distribution against itself |
| R5 | GPcounts wrapper | Install dry run against `requirements-pegasus.txt`; one- and two-sample calls on a smoke cell |
| R6 | LonGP attempt | Depends on Decision 6 |
| R7 | Measurement pilot on Pegasus, then the final submission | Per-stage resources for each regime level; cost table above replaced with measured numbers |

R1–R5 are local code and tests; none of them touches the main sweep. R7 can run while the
main sweep is still going, as long as R2–R4 needed no library change.
