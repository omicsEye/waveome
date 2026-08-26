# FINDINGS.md — empirical findings uncovered during implementation

Running log of things discovered while implementing `IMPLEMENTATION_PLAN.md` that
aren't obvious from the plan itself — surprises, validated/refuted assumptions,
robustness issues. Not a decision document: the tracker and the maintainer decide
what (if anything) merits a line in the manuscript. Newest entries at the bottom of
each task's section.

---

## T1 — refit ΔBIC + deviance explained

### Rare, non-reproducible fatal numerical fault under many sequential fits
Running `calc_feature_importance_components` across hundreds of model fits in one
long-lived process occasionally hits one of two failure modes, both apparently
low-probability and not tied to a specific input:
- A silent non-finite (NaN) result with **no exception raised** — undetectable by
  `try/except`; only catchable by validating `np.isfinite()` on the output.
- A fatal C++-level `CHECK`/`CheckNumerics` abort that **kills the whole process**
  — uncatchable in Python at any level.

A depth-isolation test (replaying the exact same fit at the exact same position in
the sequence) failed to reproduce either fault deterministically, and one crash
recurred at a *different*, earlier depth than where it first appeared. This rules
out a clean "accumulates after N fits" story in favor of "roughly constant
per-fit probability of a rare race," most likely TensorFlow `@tf.function`
retracing/graph-caching internals. Mitigated at three levels: a fail-fast
`ValueError` when the full model's own BIC is non-finite
(`calc_feature_importance_components`, `waveome/utilities.py`), a bounded
refit-with-retry for individual components, and process-level isolation
(subprocess-per-batch with batch-level retry) for any ad-hoc/scratch driver code
— mirroring the `@ray.remote(max_calls=1, max_retries=5)` pattern already in
production in `penalized_optimization` (`waveome/model_search.py`).

### Systematic positive bias in log_bf for null components under naive refit
Refitting a reduced model from a horseshoe-shrunk-but-nonzero warm start retains
residual likelihood flexibility that BIC's integer parameter-count penalty doesn't
fully offset, biasing null components' log_bf upward. A DIC-style effective-df
correction is theoretically appealing for strong-signal components but produces
NaN at the null boundary. Resolved with a clamp/refit hybrid: components already
below `VAR_CUTOFF_DEFAULT` skip the refit and are evaluated via a likelihood
clamp to `COMPONENT_CLAMP_VALUE` (safe specifically because the rest of the model
was already optimized as if the component didn't exist); components above the
floor still get the full refit.

---

## T2 — empirical-null + Benjamini-Hochberg

### Empirical-null log_bf distribution appears invariant to kernel type
The frozen decision requires stratifying the empirical null **per (kernel,
covariate) pair**, on the assumption that different kernel/covariate types
produce different-looking null evidence-score distributions (so pooling them
would miscalibrate FDR). Tested this directly across three known-null GP
simulations (seed 9102, two additive components per replicate, each
independently null by construction, leave-one-out empirical p-values):

| Simulation | Component A | Component B | Null median (A / B) |
|---|---|---|---|
| Run 1 (N=300) | Matern12, continuous covariate | Categorical, 4 levels | -1.70 / -1.70 |
| Run 2 (N=150) | Matern12, continuous covariate | Categorical, 15 levels | -1.70 / -1.70 |
| Run 3 (N=250) | SquaredExponential (stationary) | Linear (non-stationary) | -1.70 / -1.70 |

All three runs — including the stationary-vs-non-stationary pairing, the most
plausible candidate for a real difference — landed on the same null-distribution
median. Realized FDR was at-or-below nominal at q ∈ {0.01, 0.05, 0.10} in every
run, for both the stratified and pooled variants, with no visible gap between
them. Likely explanation: the T1 clamp/refit hybrid already routes any
sufficiently-collapsed component through the same clamp/BIC-penalty mechanism
regardless of kernel family, which homogenizes null behavior across types more
than the frozen decision's threat model assumed. This was checked only on small
(N=60/replicate) simulated data with three kernel pairings — real iHMP data
(different sample sizes per subject, missingness, taxonomic covariates with
skewed cardinality, NB/ZINB likelihoods) has not been checked and could behave
differently.

**Pros / cons of pooled vs. stratified, given the null distributions look the same:**

| | Pooled | Stratified (frozen decision) |
|---|---|---|
| **Pro** | More statistical power at strict q (larger combined null pool → finer p-value resolution; e.g. Run 1 pooled reached discoveries at q=0.01 that per-pair stratification couldn't, because ~200 null draws per stratum caps the achievable p-value/q-value floor above 0.01) | Matches the pre-registered/frozen methodology — no need to defend a deviation to reviewers |
| **Pro** | Robust when a given (kernel, covariate) pair has too few known-null replicates in real data to form a stable per-pair null (stratified raises a hard error in that case — `empirical_null_bh` in `waveome/utilities.py`) | Automatically correct if a *not-yet-tested* kernel/covariate combination (e.g. Periodic, Polynomial, real iHMP covariate structure) turns out to have a genuinely different null distribution — no dependence on this specific finding continuing to hold |
| **Con** | If the invariance finding is wrong for some untested kernel/covariate/likelihood combination, pooling would silently miscalibrate FDR for that pair specifically, with no per-pair diagnostic to catch it | Coarser p-value resolution per stratum (fewer null draws → higher achievable-q floor) costs power exactly at the strict end (q=0.01) |
| **Con** | Deviating from the frozen decision without much broader validation (more kernel types, real data, other likelihoods) would be redesigning locked methodology | More bookkeeping: many small per-pair groups risk near-empty null pools for rare (kernel, covariate) combinations in real data |

**Recommendation:** keep stratification as implemented (matches the frozen
decision; this is not a case for silently redesigning it). The invariance finding
is worth a line in the tracker/manuscript as a secondary robustness note, and
pooled-vs-stratified could be reported as a sensitivity check on the real iHMP
run, but stratified stays the primary rule pending broader validation.

---

## T4 — collinearity input check

### Within-unit collinearity does not imply the covariates are non-identifiable
The check originally centered each continuous covariate by its own unit mean
before computing pairwise correlation, on the theory that within-unit collinearity
is what actually breaks a longitudinal model's ability to separate two effects.
Applying this to the real iHMP covariate set flagged `study_days`, `age`, and
`time_from_max` as ~perfectly within-unit collinear (r≈1.00) — expected, since all
three are affine reparametrizations of the same per-participant clock (age
progresses at a fixed 1/365 rate, time_from_max is a fixed per-participant offset
from study_days).

The within-unit check is nonetheless the wrong diagnostic for whether the model
can tell these apart: a GP fit over the *pooled* population (not a strict
fixed-effects model) can still exploit *between*-participant variation, provided
there's enough spread in it. Concretely, participants' flare timing was staggered
across ~191 days (std) of the ~882-day study period, giving a pooled (uncentered)
correlation of only r=0.38 between `study_days` and `time_from_max` — real,
usable identifying information that a within-unit-only check discards. Switched
`_pooled_correlations` (`waveome/model_search.py`) to plain across-the-dataset
correlation instead of unit-centered: it still halts when there's truly no
distinguishing information anywhere (verified against the original synthetic
collinear-pair test, which has no between-unit spread and still halts), but no
longer flags cases where between-unit spread makes the covariates separable in
practice. On the real iHMP data, this dropped the halt/warning entirely — all
pairwise pooled correlations among `age`, `study_days`, `time_from_max`, `hbi` are
below even the moderate-warning threshold.

---

## Applying T0–T4 to real iHMP data

### Hardcoded plotting/selection thresholds calibrated to the old statistic go stale silently
The existing `ihmp_waveome.ipynb` used `plot_heatmap(metric_cutoff=41, ...)` to
show "the top 10" HBI-associated metabolites, and `plot_feature_metrics(top_n=50)`
similarly, both tuned by hand against the deprecated no-refit statistic. After
refitting under the current codebase, only 1 metabolite exceeded 41 (new log_bf
distribution: median 0.1, max 42.4) — `plot_heatmap`'s clustermap assertion
(`N>1` required) failed with no indication the *threshold* was the problem, not
the model or the data. Any notebook with a hardcoded magnitude cutoff carried
over from before this revision should be checked, not just this one — the
refit-based log_bf is not on the same scale as the old plug-in statistic, by
design (it corrects the systematic bias described under T1).

### Empirical-Bayes significance should be scoped to the covariates actually being tested, not every kernel term present
`build_component_df` collects a row per (metabolite, kernel component), which
includes structural/adjuster terms (`participant_id`, `age`, `study_days`,
`site_name`, `race`, `sex`, `general_wellbeing`) and a `"constant"` placeholder
for metabolites whose model collapsed to no signal at all (always exactly
`log_bf=0.0` by construction — see the single-kernel branch of
`calc_feature_importance_components`). Stratifying `calc_hardened_eb_qvalues`
across *all* of these crashed on `"constant"` (zero negative values, no null SD
to estimate) even though the group we actually care about (`hbi`,
`time_from_max`) had plenty. Fixed by restricting the `groups` passed to the two
research-question covariates specifically, in the notebook. This is a
now-obvious API gap worth remembering if `calc_hardened_eb_qvalues` is ever
called elsewhere without pre-filtering to the intended covariates first.

---

## Full-model fit stability: restart initialization

### The full-model optimization landscape is more multimodal than 5 restarts reliably tame
Investigating why `time_from_max` came back with zero significant metabolites
(see above), a `penalization_factor` (horseshoe scale τ) grid sweep on a small
subset — motivated by "is the horseshoe too strong?" — found no evidence for
that (the strongest `time_from_max` candidate's log_bf was *highest*, not
lowest, at the current default τ=1.0). But it surfaced a different, more
consequential problem: **independent refits of the same metabolite at the same
τ, each already using `num_restart=5`, converge to qualitatively different
models** — not just different parameter values, but different *sets* of
surviving covariates after horseshoe pruning. Three reps of one metabolite
(`C8p_QI207`, τ=0.1) gave log_bf of -14.6, +4.2, and -0.0 for `time_from_max`;
the -14.6 rep had found a 9-component kernel (several demographic covariates
surviving at once) with the *best* overall likelihood, while the sparser,
worse-fitting reps disagreed with it substantially. `num_restart=15` (vs. 5)
did not clearly fix this in the reps checked.

### Isolated the source: the full-model fit, not the per-component refit
Held one robustly-fit (`num_restart=5`) full model completely fixed and
repeated just the per-component refit (the `adam/gradient` optimization inside
`calc_feature_importance_components` that produces the reported `log_bf`)
10 times with different seeds — both the library's existing single-refit
behavior and a manually restart-wrapped version (selecting by the *reduced*
model's own likelihood, not by whichever `log_bf` looked best — that would
bias the test). Both gave **zero variance** across all 10 trials. So the
instability lives entirely in the full-model fit's choice of local optimum,
not in the refit computation, which is deterministic given a fixed input.

### `randomize_params` treats every parameter identically regardless of role — this is likely the mechanism
`PSVGP.randomize_params` (`waveome/model_classes.py:194-245`) draws *every*
trainable parameter from `N(0,1)` in unconstrained space, then pushes it
through that parameter's bijector — the same treatment for a lengthscale, a
horseshoe-penalized kernel variance, and an NB dispersion parameter, despite
these having very different sensible scales. For a softplus-constrained
lengthscale, this concentrates restarts around `softplus(0) ≈ 0.69` — a short,
"rough function" starting point for standardized covariates where genuine
smooth trends should live on an ~O(1-several) scale. `MultiOutputPSVGP`
(out-of-scope multi-output class) already uses `LogNormal(1.0, 0.5)` for
exactly this reason, just never ported to the single-output path.

### Tested smarter initialization; a naive version of the idea backfires
First attempt: keep the `LogNormal(1.0, 0.5)` lengthscale fix, and additionally
initialize kernel *variance* by sampling directly from its own already-attached
horseshoe prior (`param.prior.sample()`, `PriorOn.CONSTRAINED` confirmed) —
principled in spirit, since it's not inventing a new assumption. This backfired:
`Horseshoe` has Cauchy-like tails, appropriate for describing final beliefs but
a bad place to *start* optimization — a single `optimize_params` call from such
a draw ran >5 minutes without converging (vs. seconds normally), and had to be
killed. Replaced with `LogNormal(0, 1)` for variance (same "modest, well-behaved
positive distribution" philosophy as the lengthscale fix, no raw-prior tail
risk) — this converged quickly and reliably.

### Validated result: lengthscale + variance smart init, tested on 2 metabolites × 3 reps each
| Metabolite | Condition | log_bf std | log_bf range | full-model ll std |
|---|---|---|---|---|
| C8p_QI207 | baseline (`num_restart=5`, default init) | 8.06 | [-14.6, 4.2] | 15.19 |
| C8p_QI207 | smart init (lengthscale `LogNormal(1,0.5)` + variance `LogNormal(0,1)`) | **0.66** | [-0.2, 1.4] | **0.54** |
| C8p_QI18 | baseline | 3.20 | [-1.3, 5.9] | 6.63 |
| C8p_QI18 | smart init | **0.12** | [5.0, 5.3] | **0.40** |

~12-27x reduction in log_bf variance and ~17-28x reduction in likelihood
variance, on both metabolites tested independently. Reassuringly, smart init
isn't converging to a different/worse answer to get this stability: for
`C8p_QI18` it reliably lands on log_bf≈5.0-5.3, matching what baseline's better
reps (5.9, 5.0) already found — just consistently instead of occasionally —
and for `C8p_QI207` it consistently finds a simpler kernel than baseline ever
found, with a clean, tight, near-zero (genuinely null) `time_from_max` result
instead of baseline's scatter (including that one bloated 9-component,
-14.6 outlier).

### Broader validation: 20 randomly-sampled metabolites at τ=1.0 (the real default)
The 2-metabolite result above was on cherry-picked worst offenders at τ=0.1.
Re-tested on 20 *randomly* sampled metabolites (not cherry-picked) at τ=1.0
(the actual default used in the real analysis), 2 conditions × 2 reps each,
tracking two metabolite-agnostic robustness metrics instead of one covariate's
log_bf (different random metabolites have different covariates of interest):
whether the surviving kernel structure matches between reps, and how much the
full-model log-likelihood differs between reps.

| | kernel-structure match rate | ll abs-diff mean | ll abs-diff median |
|---|---|---|---|
| baseline | 25% (5/20) | 10.66 | 7.73 |
| smart init | 50% (10/20) | 6.14 | **1.19** |

The improvement is real and consistent — never worse on this random sample,
and the median ll difference drops ~6x (smart init is a clear win for the
*typical* metabolite) — but it is **not a complete fix**: kernel-structure
agreement only doubles, and 4-5 of the 20 metabolites (`C18n_QI31`,
`C8p_QI10`, `HILp_QI3770`, `HILp_TF51`) still show large (20-26) likelihood
disagreement between reps even with smart init. That residual instability
looks like the same underlying multimodality problem this whole
investigation started from (matches the earlier finding that `num_restart=15`
didn't fully resolve it either) — orthogonal to initialization scheme, and
out of scope for this fix specifically.

**Decision: wired in as `smart_init` in `randomize_params`, defaulting to
`True`.** A strict, consistent improvement with no observed downside
justifies changing the default rather than requiring opt-in. The residual
instability for a subset of hard-to-fit metabolites is a real, separate
concern worth flagging to the maintainer as a follow-up (e.g. a per-metabolite
fit-stability diagnostic in the analysis pipeline), not something this change
was expected to fully solve. NB dispersion (`alpha`, also positive-constrained,
no prior attached) remains an untested further candidate.

## T2: `calc_bic` computes AIC, not BIC — flagged, not yet fixed

`waveome/utilities.py:89-107` (`calc_bic`) is named and documented as BIC but
its returned expression is `2*k - 2*loglik` (AIC's formula) — a `k*np.log(n)
- 2*loglik` line (real BIC) is present but commented out, and `n` (accepted
as an argument) plays no role in the value actually returned. Confirmed via
direct code read, not just docstring inspection.

This function is used in three places, not just the significance-testing
path: `calc_feature_importance_components`'s `log_bf` (see below), and two
independent "pick the best model" scoring sites (`model_fitting.py:353`,
`model_search.py:2447`, both storing the result in a variable literally
named `bic` and presumably using it to rank/select candidate models). Fixing
the formula would change model-selection behavior at all three sites, not
just `log_bf` — true BIC penalizes complexity noticeably harder than AIC
whenever `log(n) > 2` (true here: real analysis has `n≈238`, `log(238)≈5.47`
vs AIC's flat 2-per-parameter), so this is a broader change than it first
appears and needs review at all three call sites, not just a one-line swap.

Separately (see the `log_bf`/empirical-null significance thread below): the
significance-testing path is being redesigned to drop the BIC/AIC-derived
`-p` penalty term entirely in favor of raw `ΔLL` with an empirically-fit
null location, which would make this specific mislabeling moot for that one
path. It remains a live, unfixed correctness issue for the other two
(`model_fitting.py`, `model_search.py`) call sites regardless.

### T2 addendum: `calc_bic` fix shipped, scope confirmed and accepted

`calc_bic` now computes true BIC (`k*np.log(n) - 2*loglik`) unconditionally,
not just on the significance-testing path. This also changes the two other
call sites flagged above:

- `model_search.py:2540`'s `kernel_test`, used by `keep_top_k`/`run_search`
  (the legacy stepwise kernel-search path) — `keep_top_k`'s `metric_diff=6`
  default was calibrated for the old AIC-shaped formula (~3 parameters of
  slack) and now means a different, sample-size-dependent amount of BIC
  evidence. Not recalibrated.
- `model_fitting.py:353`'s `kernel_test_reg`, used by
  `regularization.cut_kernel_components` — also affected, also not
  recalibrated.

Confirmed neither site is reached by the real-data pipeline: `penalized_optimization`
prunes via `model_classes.cut_kernel_components`, a pure variance/lengthscale
threshold with no `calc_bic` call, so the reported iHMP results are unaffected.
Both sites remain live, exported library API (`run_search`,
`regularization.cut_kernel_components`), so a future caller of the legacy
stepwise path would see the recalibration-needed behavior change described
above. Decided to document and accept this rather than recalibrate
`metric_diff` or touch `model_fitting.py`/`regularization.py`, since that
would be a new modeling judgment call (what pruning aggressiveness is
correct under true BIC?) outside this revision's single-output
significance-testing scope.

## T2 addendum 2: SE-kernel lengthscale collapse produces spurious significance — fixed

Discovered while picking representative example metabolites for the notebook:
the top-ranked squared-exponential (SE) kernel hits for `hbi`/`time_from_max`
included visibly noisy, spiky posterior predictive curves. Root cause:
nothing prevented a fitted lengthscale from collapsing arbitrarily close to
zero, at which point the SE term behaves almost like an independent
per-observation offset — free to fit noise — while `calc_bic`/`calc_metric`
still only charges it 2 parameters (variance + lengthscale), regardless of
how much effective flexibility that collapse actually buys.

Audited all 4 significant SE-kernel components in the real fit
(`fit_penalized_models_revision_full_scipy.pkl`) against the median
nearest-neighbor gap between adjacent observed covariate values (the
resolution the data can actually support):

| metabolite | covariate | lengthscale | vs. typical gap |
|---|---|---|---|
| gabapentin | hbi | 0.700 | 1.8x **longer** |
| metronidazole | hbi | 0.144 | 2.7x shorter |
| urate | hbi | 0.0296 | 13x shorter |
| proline | time_from_max | 0.000019 | 1,130x shorter |

Only gabapentin's lengthscale exceeded the data's resolution; the other
three were fitting well below it.

Three fix candidates were investigated and empirically tested against these
4 real components (proper `random_restart_optimize`, num_restart=3,
smart_init, seed=9102 — not a single continuation-refit, which for one
candidate gave a nonsense converged-looking result that turned out to be an
`ABNORMAL_TERMINATION_IN_LNSRCH` line-search failure, caught via the
`opt_status`/`opt_message` diagnostics added earlier this revision):

1. **LogNormal(1.0, 0.5) prior on lengthscale** (matches
   `MultiOutputPSVGP`'s existing, already-documented choice for the same
   reason). Refitting with this prior: gabapentin's signal survived nearly
   unchanged (log_bf 39.6→36.6); all 3 collapsed cases flipped to
   non-significant (metronidazole 6.9→-3.4, urate 10.1→1.6, proline
   9.0→-5.8, deviance_explained 65.3%→9.5%). A follow-up full-pipeline
   integration test (fresh kernel search, not just refitting one term) was
   even more decisive: urate's SE[hbi] term didn't survive pruning at all
   (replaced by a milder Lin[hbi], log_bf=6.7), and proline's
   SE[time_from_max] was pruned away entirely — no residual signal once
   the lengthscale can't collapse.
2. **Data-driven lengthscale floor** (component's lengthscale must exceed
   the median nearest-neighbor gap between observed covariate values —
   the mirror-image lower-bound analog of the existing upper-bound filter
   `keep_kernel_lengthscale_`, `waveome/utilities.py:1655`, which rejects
   lengthscales *larger* than 3x the input range). Cleanly separates
   gabapentin (passes) from the other 3 (fail), with no refit needed.
   Not implemented this pass — the prior (option 1) already resolves the
   concrete cases and was the direction chosen.
3. **Effective-degrees-of-freedom BIC penalty** (charge more than 2
   parameters for a collapsed-lengthscale SE term, proportional to its
   actual fitting flexibility). A rough ridge-regression-style proxy
   confirms the mechanism directionally (proline: ~131 effective
   parameters out of 238 observations, vs. 2 currently charged) but a
   rigorous version needs a proper derivation for the sparse-GP +
   negative-binomial-likelihood setting used here — substantially more
   work than options 1-2, and the option liked least.

**Decision: shipped option 1.** `PenalizedGP.set_lengthscale_prior()`
(`waveome/model_classes.py`, mirrors the existing `set_penalization_factor`
horseshoe-on-variance pattern exactly — same `parameter_dict`-scan
mechanism, same automatic gpflow loss incorporation via `Parameter.prior`)
sets this prior on every kernel lengthscale, called unconditionally from
`PenalizedGP.__init__` alongside `set_penalization_factor`. No changes to
`calc_bic`/`calc_hardened_eb_qvalues`/`get_significance_table` were needed —
this fixes the fitted models feeding into them, not the significance-testing
formulas themselves.

**Status: resolved.** Full 564-metabolite re-run completed
(`fit_penalized_models_revision_full_scipy_ls_prior.pkl`; pre-prior baseline
kept as `..._no_ls_prior.pkl` for comparison) in 78.45 min -- essentially
identical wall-clock to the pre-prior run, confirming the earlier timing
concern from a small (4-metabolite) integration test was Ray-overhead noise,
not a genuine slowdown. Final significance counts, stratified per (kernel,
covariate):
- `hbi`: 23 -> 15 significant. All 3 previously-flagged collapsed-lengthscale
  SE hits (metronidazole, urate) dropped out; gabapentin (the 1 genuine SE
  effect) survives essentially unchanged (log_bf 39.6 -> 35.7). The other 14
  are all `lin`-type, unaffected by this fix (no lengthscale parameter).
- `time_from_max`: 1 -> 0 significant. Proline's SE[time_from_max] term,
  the sole previous hit, is now pruned away entirely rather than surviving
  with a spuriously high log_bf -- direct confirmation it was a
  lengthscale-collapse artifact, not a real effect. The new top-ranked
  (still non-significant) candidate, C8p_QI18 (log_bf=0.2, q=0.70), is
  genuinely null-level -- no near-miss worth flagging.

Notebook example metabolites updated accordingly: docosahexaenoate (`lin`,
q=4.5e-4) for the cross-sectional HBI illustration, and C8p_QI18 kept as
the (explicitly non-significant) top time_from_max candidate for
transparency.

## T2 (continued): `log_bf` / empirical-null significance calibration — full investigation

Triggered by a user question after reviewing the real iHMP fit's significance
results: "why does the null distribution of `log_bf` stack around -2 instead
of 0?" That question unraveled into a multi-day investigation of whether
`calc_hardened_eb_qvalues` (the T2 empirical-null significance fallback) is
correctly calibrated. Documenting the full arc here since several approaches
were tried, tested on real data, and rejected for concrete, quantified
reasons — useful for a reviewer response even where nothing has shipped yet.

### 1. Why the null doesn't center at 0: `log_bf = ΔLL - p`, not a real Bayes factor

Derivation (independently re-verified by a second, fresh read of the code):
substituting `calc_bic`'s actual formula (`2k - 2·LL`, i.e. AIC, see above)
into `log_bf = -0.5·(BIC_full - BIC_reduced)` gives, exactly:

```
log_bf = ΔLL - p          ΔLL = LL_full - LL_reduced
                            p  = k_full - k_reduced (params lost when the
                                 component is dropped: 2 for squared_
                                 exponential [variance+lengthscale], 1 for
                                 lin/categorical [variance only])
```

Verified caveats on `p`: the clamp-branch shortcut (component's fitted
variance already below `VAR_CUTOFF_DEFAULT=1e-8`) hard-codes `p=1`
regardless of kernel type (`utilities.py:918`) — but real iHMP components
never hit this branch (confirmed: 45/45 sampled real components had
`clamp_used=False`), so it doesn't matter in practice. A single
(non-additive-sum) kernel's reduced model substitutes `Constant()` rather
than removing the term, giving `p=1` for a lone SE and `p=0` for a lone
lin/categorical — a real, reachable exception, but rare given the analysis
always includes an additive unit/covariate structure.

Naive Wilks-theorem intuition (`E[ΔLL]≈p/2` under an unconstrained refit)
predicts `E[log_bf]≈-p/2`. The empirically observed center is closer to
`-p` (not `-p/2`): both the horseshoe prior and, per a controlled test, even
a **near-flat/unpenalized fit** drive a genuinely null component's fitted
variance to ~0 before the drop-one comparison ever happens, leaving little
of the Wilks overfitting gain to detect. Real-data spot check (6 real
components, `fit_penalized_models_revision_full.pkl`): SE-kernel null-like
terms cluster around -0.9 to -2.3 (`p=2`), lin/categorical null-like terms
around -0.2 to -4.4 (`p=1`) — consistent with `-p` plus real per-metabolite
heterogeneity, not a fixed constant.

### 2. Quantified: the current `sigma_null` estimator is inflated

`calc_hardened_eb_qvalues` fold-and-correct recipe (`neg = -log_bf[log_bf<0]`,
`sigma_null = std(neg)/sqrt(1-2/pi)`) is only valid if the fold threshold (0)
equals the true null mean — the classical half-normal correction. Since (1)
shows the true mean is near `-p`, not 0, this assumption is violated by
construction.

Quantified via truncated-normal method-of-moments fit (fitting `(μ,σ)`
jointly from the observed `log_bf<0` sample, using the real asymmetric
truncated-normal moment equations rather than assuming truncation=mean) on
6 real `(kernel,covariate)` strata (`fit_penalized_models_revision_full.pkl`,
200-metabolite sample): **every stratum's current `sigma_null` was
inflated, never deflated**, median **1.29x**, range 1.10x-1.59x. An inflated
`sigma_null` → smaller z-scores → larger p-values → fewer significant hits
than the data actually supports. This is very likely a real, unintended
contributor to the "almost no significant `time_from_max` metabolites"
pattern that motivated the whole session's earlier work.

### 3. Why `log_bf`'s `-p` term can be dropped entirely (not just relabeled)

Since `p` is constant within a `(kernel_type, covariate)` stratum (given #1's
caveats don't materialize on real data), and since any location-fitting fix
to `calc_hardened_eb_qvalues` needs to estimate `μ` from data regardless:

```
log_bf - μ_log_bf = (ΔLL - p) - (μ_ΔLL - p) = ΔLL - μ_ΔLL
```

`p` cancels exactly. So switching the reported/tested statistic from
`log_bf` to raw `ΔLL` is provably equivalent for every downstream
significance call, *provided* location is properly fit (not assumed at 0).
It's a genuine simplification anyway: it removes the entire `p`-accounting
surface (the clamp-branch hard-coding, the single-kernel edge case, and by
extension this section's own AIC/BIC mislabeling concern) from the
significance-testing path.

Foundational note on why *some* correction (whether `-p` or an empirically-
fit null) is unavoidable: raw `ΔLL` for a nested-model in-sample comparison
is guaranteed `≥0` at the true optima of both models (the reduced model is a
literal special case of the full model) — an unpenalized `log_bf=ΔLL` would
never be negative, giving no signal to distinguish real components from
null ones, and specifically breaking the fold-based empirical-null approach
(nothing negative to fold).

**Caveat discovered via a sharp user question:** `ΔLL` empirically *does* go
negative in this pipeline, contradicting the "always ≥0" theoretical
guarantee. Root causes, both real and independently plausible: (a) the full
model and the reduced-model refit use different optimizers with different
iteration budgets (`scipy` vs a warm-started `adam/gradient`), so neither is
guaranteed to reach its true optimum, and the refit can occasionally land in
a better basin than the full model's own fit; (b) `log_posterior_density`
for a sparse variational GP is an ELBO (a lower bound), not exact
likelihood — bound tightness depends on how well the variational family
approximates each model's posterior, and that tightness need not respect
the nesting order, so `ELBO_reduced > ELBO_full` is structurally possible
even with perfect optimization of each. (a) was directly tested and found
negligible in isolation (adam vs scipy refit-optimizer swap: mean `log_bf`
diff -0.026 across 42 real components) — but that only tests refit-optimizer
*choice*, not the full-vs-reduced achieved-quality asymmetry as a whole, so
(a)+(b) combined remain the working explanation for the negative tail.

### 4. Two failed(ish) attempts at properly fitting the null location

**Attempt A — joint `(μ,σ)` truncated-normal MoM fit**, per stratum, via
`scipy.optimize.fsolve`/`least_squares` on the real asymmetric truncated-
normal moment equations. Concept validated (section 2's numbers come from
this), but **not identifiable in general**: for one real stratum
(`lin×age`, n_neg=8), the unconstrained solver converged to `μ_hat=+6.96,
σ_hat=5.36` — verified numerically to be a *genuine* alternate root
(reproduces the observed truncated mean/variance almost exactly) despite
being physically nonsensical (a positive null center). Constraining `μ≤0`
fixes the sign but then 3 of 4 testable strata collapsed onto the `μ=0`
boundary rather than finding an interior solution — small-sample
instability (available `n_neg` per stratum in real data: 8-22), not a
tuning problem. **Verdict: too fragile for production at this pipeline's
real per-stratum sample sizes.**

**Attempt B — Self & Liang (1987) boundary-parameter asymptotic theory,
"out of the box."** Testing whether a kernel's variance is zero is a
textbook boundary-of-parameter-space problem; under standard regularity,
`2·ΔLL ~ 0.5·δ₀ + 0.5·χ²₁` (a mixture of a point mass at 0 and a chi-square),
giving a closed-form p-value needing *no* empirical fitting at all —
`p = 0.5·P(χ²₁≥2ΔLL)`. Exact for lin/categorical (pure single boundary
parameter). For squared_exponential specifically, dropping the term also
loses `lengthscale`, a nuisance parameter unidentified under the null
(H0: variance=0 ⟹ lengthscale meaningless) — the Davies (1977, 1987)
problem, not plain Self-Liang; using plain Self-Liang there is a known,
one-directional (anti-conservative) approximation.

Tested against real data (`lin×hbi`, n=54, the case where Self-Liang should
be *exact*): gave 23 significant hits at q<0.05, vs. 7 (current) and 12
(Attempt A). Diagnosed precisely: pure Self-Liang theory implies
`SD(ΔLL)≈0.558` (verified via direct simulation); the empirically-fit
spread for the same stratum was `≈2.295` — **~4x wider than pure asymptotic
theory predicts**. Consistent with section 3's optimizer/ELBO-noise finding:
textbook boundary asymptotics assume exact likelihood and exact convergence,
neither of which fully holds here, so the theoretical null is too narrow and
overstates significance. **Verdict: not safe to use unmodified.**

Estimating the full Davies correction for squared_exponential (deriving the
score/profile-likelihood process over `lengthscale` for this specific
ELBO-based sparse-variational-GP model, estimating its local "roughness" per
component, and validating via simulation) was scoped as real, open-ended
statistical-methods research — likely multiple weeks with real risk of not
converging, not a bounded library task, and out of scope for this revision.
(Also found, in passing: a May-2026 arXiv preprint,
"Asymptotics for likelihood ratio tests of boundary points with singular
information and unidentifiable nuisance parameters," addresses almost
exactly this combination and explicitly calls out kernel-variance testing —
but it is an unreviewed preprint, not citable as methodological support;
the underlying Self & Liang 1987 / Davies 1977, 1987 literature it builds on
is solidly peer-reviewed.)

Also considered and set aside: Sellke, Bayarri & Berger (2001) / Berger &
Sellke (1987) universal p-value-to-Bayes-factor calibration bounds
(`B(p)≥-e·p·log(p)`) — real, well-established literature, but the wrong
direction for this problem. The bound converts an *already validly
computed* p-value (from a known, correctly-specified null sampling
distribution) into a worst-case Bayes-factor bound; it doesn't manufacture
a valid p-value from an arbitrary evidence statistic, which is exactly the
problem here.

### 5. Current best candidate: Self-Liang shape + one empirically-fit scale parameter

Hybrid, not yet decided on: keep the Self-Liang mixture *shape* (theoretically
motivated, matches the boundary-testing structure of the problem) but treat
the ~4x extra spread found in Attempt B as an additional, independent noise
term rather than trying to rescale the chi-square itself (which can't
produce the negative `ΔLL` values real data actually shows, since a scaled
chi-square is still non-negative):

```
ΔLL_observed = ΔLL_theory + eps
  ΔLL_theory ~ 0.5·point-mass-at-0 + 0.5·(χ²₁/2)     [Self-Liang shape]
  eps        ~ N(0, τ²)                                [pipeline noise]
```

`τ` is fit the *same* way the current (flawed) method already estimates its
one parameter — fold negative `ΔLL` values, correct by `1/sqrt(1-2/pi)` —
so it needs no more data than the current method already requires (works
down to 2 negative values, vs. Attempt A's 8+). No closed form for the
convolution, so p-values are computed empirically against a large simulated
null pool, matching the codebase's existing `calc_empirical_pvalue`
convention (`p=(1+#{null≥obs})/(1+B)`).

Tested on the same 13 real strata: lands between the current method and
Attempt A everywhere tested (`lin×hbi`: 7→9, vs. Attempt A's 12 and
Attempt B's 23), computes on *all* 13 strata (vs. Attempt A's 4), and agrees
closely with other methods on the one stratum that's obviously almost all
real signal (`participant_id`: 166-167 across all four methods). No
identifiability failures observed. Most promising candidate so far, but this
is a same-day prototype (single 200-metabolite subsample, no restart-
protected refit, no independent simulation validation of `τ`'s calibration)
— not validated enough to wire into the library yet.

### Status

Sections 1-5 above (truncated-normal MoM, plain Self-Liang, and the
Self-Liang+noise hybrid) were investigated and explicitly **not** adopted —
each failed either practically (small-sample identifiability) or its own
validation check (LOO calibration test on real held-out data, see below).
Reconsidered the overall strategy at that point rather than continuing to
iterate on a bespoke parametric null; decided to ship the smallest,
already-fully-diagnosed fix now and treat the bigger parametric-vs-
permutation question as separate.

**Shipped:** `calc_hardened_eb_qvalues` (`waveome/utilities.py`) gained a
`null_offset` parameter (scalar or per-observation array; default `0.0`,
fully backward compatible) that corrects both the location (fold/center
point) and, because the `sqrt(1-2/pi)` scale correction is only valid when
the fold point equals the true mean, the resulting `sigma_null`'s scale —
not just where p-values are centered. Callers pass `-p` per component
(kernel-type-dependent: 2 for squared_exponential, 1 for lin/categorical),
computed per-observation rather than per-stratification-group since the
real notebook groups by `covariate` alone (not `(kernel, covariate)` as
frozen decision 4 specifies — a pre-existing gap, noted but not addressed
here) and kernel type therefore varies within a group.

Wired into `examples/iHMP/ihmp_waveome.ipynb`: `build_component_df` now
attaches a `null_offset` column per component; both significance call
sites (cross-sectional HBI heatmap, main Significance section) pass it
through. Verified end-to-end against real fitted data
(`fit_penalized_models_revision_full.pkl`, the Jul 22 pre-restart-protected
fit — illustrative only, not a final result): `hbi` sigma_null 2.932→2.545,
pi0_hat 0.958→0.679, significant metabolites at q≤0.1 22→40; `time_from_max`
sigma_null 3.018→2.539, pi0_hat and n_sig unchanged at 1.000/0 — the fix
recovers real power where the corrected calibration supports it and stays
appropriately null where it doesn't, rather than inflating hits
indiscriminately.

**Deliberately not addressed by this fix** (anchoring at the theoretical
`-p` rather than empirically fitting location was chosen specifically to
avoid the small-sample instability from section 4's Attempt A): this does
not use a data-fit location, so it inherits whatever bias exists between
the true per-stratum null center and the theoretical `-p` anchor (real
data showed centers ranging roughly -0.75 to -3.4 around nominal `-p` of
-1 or -2 in section 1's spot check). It also does not address the
`squared_exponential`-specific Davies/nuisance-parameter concern (section
4, Attempt B) or the still-unresolved question of whether the notebook's
per-covariate-only stratification (rather than per-`(kernel,covariate)`)
should be fixed to match frozen decision 4.

**Update (2026-08-02): G1 is resolved.** Per `waveome_revision_tracker.md`
item 5, the combined refit×permutation compute estimate (≈5 covariates ×
200 perms × ~32-min full run ≈ 530+ core-hours *before* reduced refits)
came back prohibitive. **Decision: T3 (permutation) will not be pursued;
T3' (hardened-EB fallback, this section's method) is the real-data
significance method**, not a documented-but-secondary fallback. That
raises the stakes on everything in this section — the calibration issues
found and the fix applied are now load-bearing for the actual reported
results, not an exploration of one option among several.

Also worth noting: the tracker's own pre-existing issue list for this
exact procedure (written before this session's investigation) independently
flagged several of the same problems found here — the boundary-null
misspecification ("Path A... abandon or state this correctly", matching
section 4 Attempt B's finding that plain Self-Liang is anti-conservative),
the symmetric-Gaussian-null misspecification (matching section 1-2's
null-centering finding), and — **flagged there as High severity and still
not fixed by this session's work** — pooling heterogeneous kernel types
into one null ("Pools heterogeneous log-BFs (ID vs SE-on-HBI vs linear)
into one null | High | Stratify FDR by kernel type"). That's the same gap
noted two paragraphs up (the notebook stratifies by `covariate` alone, not
`(kernel, covariate)`) — independently corroborated as a real, high-priority
item, not just an incidental observation from this investigation.

## T3'' — the significance null, rebuilt (2026-08-05 → 08-07)

Continues the section above. Everything below was driven by a crash in
`calc_hardened_eb_qvalues` on the first no-prune fit, which turned out to
be the visible end of two separate numerical faults and then of a
methodological dead end. Recording the arc because most of it is
negative results that a reviewer response can cite.

### 1. Root cause of the non-converged fits: a Horseshoe gradient underflow

74% of no-prune full-model fits (417/564) reported
`ABNORMAL_TERMINATION_IN_LNSRCH` with 0 iterations. Root cause: once a
constrained kernel variance falls below ~8e-78, `tfp.Horseshoe.log_prob`'s
**gradient** underflows to exactly `-inf` while its *value* stays finite,
so nothing raises — the whole gradient vector is poisoned and L-BFGS-B
fails its first line search. Perfectly separating: every non-converged
model had a variance in that zone, every converged model did not. It
produced `log_bf` values up to 26 million.

Not GP- or kernel-specific — a pure float64 limitation of the TFP
distribution. **Fixed** by flooring every positive-constrained parameter
(`set_variance_floor`, `VARIANCE_FLOOR = 1e-10`, via
`gpflow.config.set_default_positive_minimum`). A full re-run took
convergence from 26% → 98.2%.

### 2. The clamp shortcut was manufacturing a point mass

With the floor in place the `calc_hardened_eb_qvalues` crash persisted,
for a different reason. The clamp shortcut (skip the refit for components
already below `VAR_CUTOFF_DEFAULT`, evaluate a clamped variance instead)
returned a near-deterministic `log_bf` — it was reading back a fixed
constant, not an optimization result. Measured spread across metabolites:
**sd 0.00004 (SE), 0.00034 (lin)**. That artificial atom is what broke
every downstream null-distribution assumption.

It also charged `k_full - 1` parameters regardless of kernel type —
correct for lin (p=1), wrong for squared_exponential (p=2) — so every
near-floor SE component sat at the lin-appropriate value (−2.6) instead of
its own (−4.80). Testing SE strata against a correctly-computed anchor
gave a spurious **564/564 "significant"**. **Removed**; all components are
now genuinely refit.

### 3. The empirical-Bayes null cannot be rescued (rejected)

With clean data, a synthesis was tried: take the null's *location* from
the near-floor population (near noise-free) and its *spread* by folding
the non-floor population's below-anchor tail. It produced defensible
counts (lin:hbi 137, SE:hbi 2, SE:time_from_max 0) but does not survive
diagnostics:

- The theoretical anchor `-p·log(n)/2` is **biased by +1.736 (p=1) and
  +0.672 (p=2)** — a null component still yields that much ΔLL because the
  reduced-model refit doesn't reproduce the full model's fit. The
  floor-derived anchor measures this directly and lands on the mode of an
  *independent* population, which is real validation of the location.
- But the **shape fails**: Shapiro rejects half-normality in **6/12
  strata**, from two opposed causes — depletion near the anchor (the
  near-floor/non-floor split removes exactly the values closest to it) and
  impossible outliers 4–5 units below it (ELBO/optimizer inversions).
- **Scale is not identified.** Three defensible estimators span up to **3×**
  (SE:study_days 0.55 / 0.68 / 1.48) and move hit counts ±25%. The
  best-*fitting* estimator is also the one yielding most discoveries — an
  unacceptable researcher degree of freedom.
- Storey's π₀ is **structurally incompatible**: the floor atom sits at
  exactly p=0.5 and `p > λ` excludes it, collapsing π̂₀ to 0.03–0.17 where
  the truth is ~0.95 and inflating hits 10–30× (547, 564, 549). Tie-aware
  π₀ saturates at 1.0, i.e. plain BH.

**Decision: abandon the parametric null.** Diagnostics in
`examples/iHMP/validate_null_visually.py`.

### 4. G1 revisited — permutation is affordable after all

G1 rejected permutation at "5 covariates × 200 perms × ~32-min full run ≈
530+ core-hours". That costed a **full run including kernel search**. But
the no-prune analysis fits **one identical kernel structure to all 564
metabolites** (verified: 1 distinct structure) — there is no per-metabolite
search, so a null draw costs one fixed-structure refit (~4s) plus
drop-one refits (~3.5s), measured. It also means there is no
post-selection-inference problem.

Measured throughput is **2.98 s/draw** (parallelism only ~3.7× on 10
cores; TF threads contend with Ray, and pinning barely helped), so the
full B=20 two-covariate run is **~19.6 h** — a maintainer HPC job, not the
530 core-hours that killed it.

### 5. Which permutation scheme (four rejected)

Tested rather than argued, on `HILp_QI2874` (participant_id log_bf 20.6,
SE[hbi] 36.6 — where subject-proxying should be maximal):

| scheme | between-frac | invented values | verdict |
|---|---|---|---|
| real data | 0.413 | — | reference |
| **within-subject shuffle** | **0.413** | 0.0% | **kept** — exact, preserves structure |
| global shuffle | 0.199 | 0.0% | rejected — anti-conservative |
| Normal (μ,σ) swap | 0.525 | 93.8% | rejected |
| empirical donor resample | 0.569 | 0.0% | rejected |

- **Global shuffle** destroys HBI's clustering (0.199 is exactly the
  chance baseline 48/237). Because an SE kernel on a *clustered* covariate
  partially proxies the subject intercept, that gain is a real feature of
  the null — deleting it made a 4.1-SD result look like **28.6 SD**.
  Kernel-specific: for `lin` the two schemes are identical (`k = σ²·xᵢxⱼ`
  builds no within-subject blocks).
- **Whole-trajectory swap** is exact but needs matched block sizes
  (this cohort happens to have n=4/5/6 groups of 16/10/14, ~10³¹
  permutations — but a library cannot assume that).
- **Normal (μ,σ) swap** — HBI is a bounded discrete score (14 distinct
  values, skew +2.03, Shapiro p=1e-16); 93.8% of sampled values are HBI
  scores that cannot exist.
- **Donor resample** (non-parametric form of the same idea) invents
  nothing, but its null went **degenerate (sd 0.00)** — more degenerate
  than the global shuffle it was meant to improve on.

Also rejected: a **within-between (Mundlak) decomposition** of hbi into
two covariates. It is well-conditioned (r = −0.000 between the parts) and
informative — C18n_QI41's `lin[hbi]`=18.0 becomes `lin_within`=5.3 with
participant_id rising −1.0 → 4.0, i.e. two-thirds of an apparent HBI
effect was subject structure — but it changes the *model* rather than the
test, doubles components per time-varying covariate, and pushes
statistical vocabulary onto users. Not adopted.

### 6. Where it landed

**Within-subject free shuffle** (free, not circular: the additive kernel
never uses hbi's time-ordering, and free gives 86 configurations/subject
vs 4.8). The observed statistic is **recomputed through the identical code
path** (zero shuffle), so optimizer/ELBO artifacts appear on both sides
and cancel — verified: 25.3→25.31, 18.0→18.03, 36.6→36.60. Empirical p
with **tie tolerance** (two components at one collapsed state, −0.9681 vs
−0.9675, otherwise got p=0.90 and p=0.31).

This tests the **within-subject** association; between-subject structure
stays in the null on both sides, so it can neither create nor be credited
as a hit. A separate subject-level permutation (49 scalars, exact, one
extra fit per metabolite, no refits under permutation) covers the
between-subject question. Both are general — neither needs matched block
sizes.

`age` is a warning case: between-fraction **1.000** yet it "varies within
subject" (age ticks up during follow-up), so a naive varies-within rule
would silently give it a meaningless within-subject test.

### 7. Open

- **Pooling.** Per-metabolite nulls differ in *scale*, not just location
  (SE:hbi per-outcome SD 0.00–18.43). Centring fixes location only, so
  pooling a shared tail is anti-conservative for wide-null metabolites.
  But per-metabolite p-values are **structurally impossible** here: BH
  rank-1 at m=564 needs p ≤ 8.9e-5, i.e. B ≥ 11,240 *per metabolite*.
  Pooling across features is standard (SAM, Storey–Tibshirani) — what
  certifies it is calibration, not theory.
- **T2 has never been run** and is now load-bearing. Staged: Stage 1 =
  p-value uniformity under a complete null (~1.5–2.5 h,
  `examples/simulations/sim_fdr_stage1_uniformity.py`); Stage 2 = the full
  realized-FDR-vs-nominal table. The simulation must reproduce unequal
  visits, a covariate with ~41% between-subject variance, horseshoe
  collapse-to-floor, and both SE and lin components, or it passes
  trivially.
- Nothing here is wired into the library; `calc_hardened_eb_qvalues`
  remains the shipped path. *(Superseded: permutation is now the shipped
  path and that function has been removed -- see 18-19.)*

### 8. Why the mixed-model variance-component literature doesn't transfer

Raised as a reviewer-anticipating question: testing `variance = 0` is a
standard problem in mixed models, so why not use that machinery? The chain
is Self & Liang (1987) → Stram & Lee (1994) → **Crainiceanu & Ruppert
(2004)**, who showed the asymptotic mixture is badly wrong in finite
samples and derived the *exact* finite-sample null of the restricted LRT →
**Greven, Crainiceanu, Küchenhoff & Peters (2008)** for zero variance
components → implemented in **RLRsim**. Penalized splines are close
cousins of GPs, so the instinct is right.

Four independent blockers:

1. **Conditionally Gaussian responses.** RLRsim states this as a scope
   condition. We fit a negative binomial to counts.
2. **A single variance component.** The exact distribution is derived for
   one component under test; our kernel carries 13, all penalized at once.
3. **It needs an exact restricted likelihood; we optimize an ELBO.** A
   variational bound's tightness differs between full and reduced models
   and need not respect nesting -- which is why `ΔLL < 0` occurs in real
   fits, structurally impossible under exact nested likelihoods.
4. **The horseshoe changes the null.** These tests assume (RE)ML. Under a
   shrinkage prior the null is governed by the prior driving variances to
   zero, which is why we measure a point mass at the collapse floor in
   ~2/3 of outcomes rather than `0.5·δ₀ + 0.5·χ²₁`.

Plus the Davies (1977, 1987) problem for SE kernels (lengthscale
unidentified under H0), which plain Self-Liang does not cover. And this is
not speculative: section 4 Attempt B tested plain Self-Liang on `lin×hbi`,
where it should be *exact*, and found theory implies `SD(ΔLL) ≈ 0.558`
against an empirical `≈2.295` -- **4x too narrow**, 23 hits vs 7.

**What we did take from it:** RLRsim's contribution is less the algebra
than the philosophy -- when the asymptotic null is wrong, simulate it
under the fitted null rather than deriving it. Generalized, that is a
parametric bootstrap (simulate from the fitted reduced model, refit both,
difference), and it is worth running on a subset as a **cross-check**: two
nulls resting on entirely different assumptions (exchangeability vs
correct specification) agreeing is much stronger evidence than either
alone. Not adopted as the primary, because it assumes the reduced model is
correctly specified where permutation needs only exchangeability, and it
plugs in MAP estimates so it understates parameter uncertainty.

### 9. Why not posterior sampling → Bayesian FDR

The model is already Bayesian, so the coherent answer would be posterior
inclusion probabilities + Bayesian FDR (tracker option C). It does not
work as posed:

- **The horseshoe places no mass at zero.** Bayesian FDR consumes
  `P(H0ᵢ | data)`; a continuous shrinkage prior gives `P(variance=0) = 0`
  exactly, so that quantity does not exist. Carvalho-Polson-Scott's
  shrinkage weight κ is a *decision heuristic* (from the posterior mean
  being `(1−κ)·y`), not a probability, so feeding it to a cumulative-mean
  FDR rule is invalid. Genuine inclusion probabilities need a
  spike-and-slab prior -- a different model.
- **There is no working sampler.** `hmc_sampling` exists in `utilities.py`
  but is never called (only a commented-out reference at
  `model_classes.py:2136`), and as written samples *every* trainable
  parameter including `q_mu`/`q_sqrt`, conflating the variational
  approximation with posterior sampling. Horseshoe posteriors also need a
  non-centered reparameterization for HMC that isn't there.
- **Cost is likely worse.** ~15,000 gradient evaluations per metabolite vs
  a few hundred for L-BFGS -- ≥60 core-hours optimistically, before funnel
  divergences, against ~72 for permutation with far less convergence risk.
- **It inherits an uncalibrated τ.** Decision 10 leaves
  `penalization_factor = 1.0` fixed and explicitly not frozen. κ and any
  inclusion probability depend directly on τ, so the verdict would be a
  function of an arbitrary constant. **Permutation is immune** -- τ is
  applied identically to observed and permuted data and cancels.
- **It doesn't answer the reviewers.** R1.M5/R2.5 asked for realized
  FDR/FWER vs nominal. Bayesian FDR controls a posterior expected
  proportion -- a different, prior-dependent guarantee -- so T2 would
  still be required on top of the modelling change.

From scratch, with a spike-and-slab prior, a working sampler and a
calibrated τ, this would be cleaner than anything in sections 5-7. Getting
there from here is a research project, not a revision task -- the same
verdict reached on the Davies correction.

### 10. T2 Stage 1: pooling fails, and the fix

Three attempts, the first two vacuous, recorded because the failure mode
is instructive.

**Attempts 1-2 tested the wrong null.** Both simulated a COMPLETE null
(no `cindex` effect at all). Under two tunings -- `categorical[id]`
collapsing in 83% then 10% of outcomes -- SE nulls stayed degenerate
(max SD 0.208, then 0.548, against 18.43 in the real cohort), and both
reported clean results (0/150 false discoveries, `frac<0.05 = 0.040`)
that meant nothing. **The wide nulls exist precisely because a
within-subject shuffle preserves the between-subject association**, so
removing that association makes the regime unreachable at any tuning. The
complete null is also stricter than the hypothesis under test: H0-within
permits a between-subject effect.

**Attempt 3 used the correct null** (`beta·cindex_between`, never a
within-subject effect) and reproduced the regime: max null SD 19.2 (lin)
and 13.5 (SE), `categorical[id]` collapsed 0.33. A **regime gate** now
prints these before any p-value so the check cannot pass vacuously again.

**Pooling a shared tail is badly anti-conservative**, and the aggregate
hides it:

| | degenerate | narrow | WIDE | overall |
|---|---|---|---|---|
| lin | 0.000 (n=99) | 0.000 (n=32) | **0.368** (n=19) | 0.047 |
| SE | 0.000 (n=102) | 0.000 (n=35) | **0.385** (n=13) | 0.033 |

7-8x nominal for wide-null outcomes, invisible in the aggregate because
the degenerate two-thirds contribute zero and dilute it. Mechanism: an
outcome whose own null has SD ~19 compared against a tail built mostly
from point masses.

**Fix, scored side by side** (adaptive B: only 52/150 outcomes needed
topping up B=20→60, since degenerate ones are self-resolving -- 2,080
extra draws instead of 6,000):

| WIDE subgroup | lin (n=19) | SE (n=13) |
|---|---|---|
| pooled | 0.368 | 0.385 |
| (a) scale-stratified | 0.000 | 0.000 |
| (b) standardised | **0.053** | **0.077** |

Both remove the inflation. (a) is conservative (0/19, 0/13, and 0.000
across every SE subgroup) and would cost real power; **(b) standardised
targets nominal** and is adopted. **Caveat: n=19 and n=13 give ±0.05 on a
0.05 proportion**, so both are *consistent with* nominal and neither is
*demonstrated* correct. Settling it needs ~M=800 outcomes (~100 in the
wide bin), roughly 14 core-hours.

### 11. Stage 1 at scale reversed the choice made at n=19

Section 10 adopted standardised pooling on wide-subgroup rates of 0.053 and
0.077 against nominal 0.05, from n=19 and n=13 outcomes. Rerunning at M=400
(B0=10 screen, B1=100 top-up; 153 of 400 outcomes non-degenerate, so 13,770
extra draws rather than 36,000) put ~60 outcomes in the wide bin and
**inverted the ranking**:

| WIDE subgroup | lin | SE |
|---|---|---|
| pooled (broken) | 0.368 (n=19) | 0.385 (n=13) |
| (a) scale-stratified, M=400 | **0.032** (n=62) | **0.028** (n=36) |
| (b) standardised, M=400 | 0.081 | 0.111 |

At the larger sample (a) sits just below nominal and (b) sits above it in
both kernels, 1.6x and 2.2x. Neither is individually significant (+1.1 and
+1.7 SE) but (b) errs the same direction in both, and that direction is the
one the whole exercise exists to remove. **The small-sample evidence would
have shipped the anti-conservative variant.**

### 12. Quantile regression replaced binning; leave-one-out was immaterial

Binning works but needs an arbitrary cutpoint, and it over/undershoots
either side of it (narrow 0.070, wide 0.032 against nominal 0.05). Replacing
it with a conditional quantile regression of the pooled centred draws on
each outcome's own null SD -- no bins, smooth in scale -- calibrates better
and gives slightly more power at identical FDR:

| lin | narrow | WIDE | q=0.01 FDR / power | q=0.05 FDR / power |
|---|---|---|---|---|
| scale-stratified | 0.070 | 0.032 | 0.010 / 0.866 | 0.029 / 0.893 |
| **quantile regression** | **0.047** | **0.048** | 0.010 / **0.884** | 0.029 / **0.902** |

**Adopted.** Caveat: it assumes the conditional quantile is linear in null
SD, which is a real modelling assumption with SD spanning 0 to 21 and is the
thing most likely to break on other data.

IHW (Ignatiadis & Huber 2016) also motivated a **leave-one-out** check --
each outcome's own draws sit in the pool it is judged against, the
circularity IHW's cross-fold design exists to prevent. Implemented and
measured: **not one p-value moved** in either dataset. Each outcome
contributes ~100 draws to a bin pool of 6,200-8,600, so removing its own
shifts the count by ~1.5%, never enough to cross a threshold. Real in
principle, immaterial at this scale.

### 13. T2 Stage 2 -- realized FDR, and GPD rejected

400 outcomes, 112 true positives (28%), within-effect sizes 0.15-1.20 so the
table has a power gradient rather than all-or-nothing detection. True nulls
still carry between-subject effects, preserving the wide-null regime.

**This is the evidence R1.M5 / R2.5 asked for:**

| q | discoveries | false | realized FDR | power |
|---|---|---|---|---|
| 0.01 | 98 | 1 | 0.010 | 0.866 |
| 0.05 | 103 | 3 | 0.029 | 0.893 |
| 0.10 | 104 | 3 | 0.029 | 0.902 |

**GPD tail approximation (Knijnenburg et al. 2009) tested and rejected.** It
matches at q=0.05 and 0.10 but **loses 80% of its power at q=0.01** (0.179
vs 0.866). Cause: 49 of 113 GPD attempts (43%) returned shape c < 0, a
*bounded* tail whose endpoint the observed value exceeds, giving sf = 0 --
not a small p-value but an invalid one. Those fall back to the empirical
p-value, capped at 1/(B+1), which q=0.01 cannot reject. The method adopted
to beat the resolution floor dumps 43% of tests back onto it. Intrinsic to
fitting a GPD to these collapse-dominated tails; more permutations would not
help.

**Known gap:** both methods show ~0 power for SE components here, because
the simulated true positives are *linear* within-subject effects that
`lin[cindex]` captures and `SE[cindex]` has no reason to. SE produces no
false positives, but **SE power is unvalidated**. Affects a power claim, not
error control. ~15h to fix.

### 14. iHMP results

16.9h, 32,358 draws, **zero failed fits** (before the variance floor, 74% of
fits failed). 399 of 1128 components (35%) had a non-degenerate null and
were topped up.

| stratum | q<0.05 | q<0.10 | min p | attainable floor | verdict |
|---|---|---|---|---|---|
| lin:hbi | 111 | 140 | 5.10e-05 | 5.10e-05 | at floor |
| lin:time_from_max | 3 | 3 | 5.39e-05 | 5.39e-05 | at floor |
| SE:hbi | 1 | 1 | 5.10e-05 | 5.10e-05 | at floor |
| SE:time_from_max | 0 | 0 | 3.24e-03 | 8.59e-05 | **above floor** |

Consistently fewer hits than hardened-EB (111 vs 137, 3 vs 9, 1 vs 2) -- the
expected direction, since permutation leaves each metabolite's
between-subject structure in its own null.

`lin:time_from_max` was re-run at B=120 because its floor cleared BH's
rank-1 threshold by only 3%. Doubling the draws moved the floor 8.59e-05 ->
5.39e-05, a 1.64x margin, and **nothing changed**: same 3 metabolites, zero
decisions flipped. The count is not an artifact of thin resolution.

### 15. Between-subject test -- the complement, and a null result

The within-subject permutation preserves each subject's own values, leaving
hbi's 41% and time_from_max's 47% between-subject variance untested. Covered
by a subject-level test: refit without the covariate's components, reduce
each subject to (mean covariate, mean Pearson residual), Spearman across the
49 subjects, permute the 49 subject-level values. Exact and fully general --
one number per subject, so subjects are exchangeable regardless of visit
count. One extra fit per (metabolite, covariate); no refits under permutation.

| covariate | q<0.05 | min p | floor | verdict |
|---|---|---|---|---|
| hbi | 0/564 | 0.0015 | 5.0e-05 | above floor -- genuinely null |
| time_from_max | 0/564 | 0.0086 | 5.0e-05 | above floor -- genuinely null |

**So HBI's association with the metabolome is within-patient.** Metabolites
track a patient's own disease activity; patients with higher *average* HBI do
not have systematically different levels once the rest of the model is
accounted for. This retires the caveat that the within-subject test leaves
41%/47% of the variance uncovered -- it is covered, and empty. Limits: n=49
subjects, so only fairly strong between-patient effects are detectable, and
it rests on residuals from the no-covariate refit.

### 16. Sizing rule, and telling "not extreme" from "could not resolve"

BH rejects the k-th smallest p if p <= qk/m, so the hardest case is rank 1
needing p <= q/m, while the smallest p obtainable from pooled draws is 1/N.
Hence:

    N_pooled  >=  m / q            i.e.   n_degen*B0 + n_nondegen*B1 >= m/q

Self-correcting: the more components collapse (contributing only B0), the
more the survivors need. Necessary but not sufficient -- the post-hoc check
is whether the smallest observed p sits **at** 1/N (more draws could change
the answer) or well **above** it (the data simply is not extreme). That
distinction is what separates SE:time_from_max, which is genuinely null,
from the other three strata.

**Ties.** 72 of the 111 lin:hbi hits sit exactly at the floor. The count is
sound but they cannot be *ranked*; a top-N list would need more draws than a
set does.

**Boundary sensitivity.** HILn_QI42 moved q = 0.0536 -> 0.0474 on a dp of
0.0014 when the pooled null was perturbed. Results within a few percent of
the threshold are not stable to small changes in the pool, and borderline
metabolites should not be presented as though the cutoff were sharp.

### 17. Rejected: prefiltering components already at the variance floor

A component whose *fitted* variance sits at the collapse floor cannot be
significant -- its observed log_bf is the minimum attainable, so excess <= 0
and p >= 0.5 for any B. Provable, and confirmed: 0 of 832 such pairs were
significant. Skipping them looked like free compute.

It is not free. **Those draws also constitute the pooled sample every other
component is judged against.** Dropping them re-fits the quantile regression
on a different sample: lin:time_from_max went 3 -> 0. Synthesising them
(their null is a point mass at the observed value -- verified, 100% within
the tie tolerance, median deviation 2.8e-8) repaired that, but repeating the
value *exactly* built a machine-precision tie mass that made the quantile
regression divide by zero; adding jitter at the measured 4e-8 scale fixed
the numerics but **one metabolite still flipped** (HILn_QI42, q 0.0474 vs
0.0536), because synthetic draws cannot be the same numbers as real ones.

**Reverted.** The saving was also smaller than first calculated: 26%, not
61% -- the first figure counted at the *component* level, but one draw serves
both components of a covariate and the top-up phase (19,950 of 32,358 draws)
is untouched by the prefilter. Three layers of correction for ~4h, still
changing an answer, is a poor trade.

### 18. Corrections to earlier entries in this document

- **SE:time_from_max was described as resolution-blocked. It is the
  opposite** -- the only stratum where resolution is *not* the constraint.
  Its pool reaches 8.59e-05, below BH's 8.87e-05, and its most extreme
  metabolite is at 3.24e-03. Its zero is a finding.
- **The prefilter saving was quoted as 61%; it is 26%** (see 17).
- The between-subject test first reported 0/564 from **200** permutations,
  flooring every p at 5e-03 -- 56x above BH's threshold, so it could not
  reject anything. Rerun with 20,000 (vectorised: Spearman is Pearson on
  ranks, so all draws are one matmul). The rule in section 16 was derived one
  message before it was violated.

- **`calc_hardened_eb_qvalues` was removed from the library**, not merely
  demoted. It was introduced on this branch (`7a0e40c`) and never shipped, so
  nothing external depends on it, and keeping an uncalibrated method exported
  as a "fallback" would amount to recommending one this document shows is
  wrong. Removed with it: `get_significance_table`'s `null_offset` column and
  `_component_param_count`, which existed only to compute it. `null_offset`
  was independently a wrong quantity -- the theoretical anchor it encodes is
  biased by +1.736 (p=1) and +0.672 (p=2), measured in section 3.
  Earlier entries in this document (sections 1-10) still refer to it; those
  are the record of what was tried and are left as written.

### 19. Where the code stands

Shipped in `waveome/`: `permute_covariate`, `calc_between_unit_fraction`,
`calc_permutation_pvalues` (utilities), `GPSearch.permutation_significance`
with adaptive B, checkpoint/resume, and a between-unit-variance warning.
`calc_hardened_eb_qvalues`, `null_offset` and `_component_param_count` were
removed outright (see 18). `calc_empirical_pvalue` and `empirical_null_bh`
are kept: currently unused, but correct, and they are frozen decision 4's
machinery for the known-null simulation path.

Three resume bugs were found and fixed (`c756656`), all from assuming a
checkpoint always matches the scope of the call resuming it -- which fails
in exactly the case checkpointing enables, topping up one covariate from a
multi-covariate run. `tests/test_permutation_resume.py` covers it, verified
red/green.

Not yet in the library: the between-subject test (script only), p/q columns
on `get_significance_table`, `plot_heatmap(significance=...)`.

## 20. SE components are ~95% numerically dead, and `deviance_explained` credits dead components (2026-08-26)

Found while choosing manuscript showcase metabolites: proline's
`SE[time_from_max]` panel shows a striking peaked curve labelled DE=8.5%
next to log_bf=-5.5. Chasing that contradiction turned up a structural
problem in two reported quantities.

### The survey

Across all 7,332 components of the reported fit, **85% sit on exactly two
log_bf values**, determined by kernel parameter count, not by data:

| log_bf | count | kernel types |
|---|---|---|
| -1.0 | 4,153 | categorical (2,283) + lin (1,870) — 1-parameter |
| -4.8 | 2,102 | squared_exponential — 2-parameter |

`deviance_explained` and `log_bf` are **uncorrelated** among live components
(spearman +0.031, n=1,782), and ~70% of components with substantial DE carry
negative log_bf at every threshold tested (DE>0.02: 67.8%; >0.10: 69.6%;
>0.20: 69.5%).

### The mechanism — the pinned components are dead

| group | n | median variance | at VARIANCE_FLOOR | median lengthscale |
|---|---|---|---|---|
| pinned at -4.8 | 2,102 | 1e-10 | **99.9%** | 2.117 |
| unpinned | 154 | 0.124 | 0.0% | 1.43 |

The pinned components have variance sitting exactly on `VARIANCE_FLOOR`
(1e-10). Their lengthscale is 2.117 for *every* covariate alike — hbi,
study_days, age, time_from_max, identical to four figures — which is
`exp(1 - 0.5^2)`, the mode of the LogNormal(1.0, 0.5) lengthscale prior
added in "T2 addendum 2". With variance at zero the lengthscale is
unidentified and reverts to its prior mode.

So log_bf = -4.8 is not "strong evidence against"; it is the constant a
dead 2-parameter component contributes. Likewise -1.0 for dead 1-parameter
components. This is *the same phenomenon as the collapse atom* that makes
permutation nulls degenerate (section 13) — seen from the parameter side.

### `deviance_explained` credits components that contribute nothing

Of the 248 SE components with DE>0.10 that are pinned at -4.8, **246
(99.2%) have variance at the floor, and their median DE is 0.969**. A
component contributing exactly nothing is credited with explaining 97% of
its model's gain over null; 81 dead components score DE=1.000.

The cause is that `calc_feature_importance_components` computes marginal DE
from the *refit* reduced model. Dropping a dead component changes the model
by nothing, but the subsequent refit lands the surviving components in a
different local optimum, and DE attributes that entire prediction shift to
the dropped component. **DE as currently computed measures refit
instability, not a component's explanatory contribution**, whenever the
component is dead.

**Correction:** an earlier draft of this section claimed proline's
`SE[time_from_max]` panel was one of these dead components. It is not. Its
variance is 0.1016, well off the floor, and its lengthscale is 1.51 (fitted,
not the prior mode) = 211 days against a ~3-day median gap between adjacent
observed values. It is one of the 154 live components, its log_bf is -5.5
rather than the -4.8 mass point, and the 2.06x peak-to-trough effect it
draws at ~112 days post-maximum is real and supported by 93 of 238
observations. Its negative log_bf has a different cause -- see section 21.

### Consequence for the reported nonlinearity result

```
SE:hbi            dead 537/564 (95.2%)   alive 27
SE:time_from_max  dead 534/564 (94.7%)   alive 30
```

log_bf is a constant for ~95% of the metabolites in both tested SE strata.
The permutation nulls, however, are **not** correspondingly absent -- the
non-degenerate counts are almost identical to the linear strata:

| stratum | live nulls | median null SD | significant |
|---|---|---|---|
| lin:hbi | 273/564 | **0.378** | 140 |
| SE:hbi | 272/564 | 1.32e-05 | 1 |
| lin:time_from_max | 109/564 | 1.84e-05 | 3 |
| SE:time_from_max | 107/564 | 1.24e-05 | 0 |

So the SE strata are *not* untestable for want of usable nulls; they have as
many as the linear strata. What separates them is null *width*: SE:hbi's
median live null is four orders of magnitude narrower than lin:hbi's. Under
permutation the component occasionally comes alive, which is what gives the
null any spread at all.

A narrow null does not by itself imply no power -- the conditional quantile
regression conditions on each metabolite's own null SD, and lin:time_from_max
produced 3 hits from a null of comparable width (1.84e-05). So **what
remains unestablished is whether 1/564 and 0/564 reflect genuine absence of
nonlinear association or insufficient power.** The observed statistic being
constant for 95% of metabolites is a reason for concern, not a proof of no
power. Do not state in the manuscript that nonlinearity was tested and found
absent until this is settled.

This is not obviously a defect. The LogNormal prior was added deliberately
(T2 addendum 2) because unconstrained SE lengthscales collapsed below the
data's resolution and fit per-observation noise — proline's was 1,130x
shorter than the median nearest-neighbour gap, and the prior moved it from
log_bf 9.0 to -5.8. Genuine nonlinearity still survives it (gabapentin's
SE[hbi], 39.6 -> 36.6). The 154 live SE components show the prior does not
kill everything. The open question is whether ~95% death is correct
suppression of terms that never helped, or over-suppression.

### What must not be said, and what is safe

- Do **not** report DE for a component without checking its variance is off
  the floor; the number is meaningless for dead components.
- Do **not** state that nonlinearity was tested and found absent. State that
  SE terms overwhelmingly collapse under the lengthscale prior, and that the
  permutation test has no resolving power where they do.
- The linear results are **not** threatened by this. `lin` components pin at
  -1.0 too, but `lin:hbi`'s live nulls have median SD 0.378 -- four orders
  wider than any other stratum -- and it produced 140 hits.

Diagnostic: `examples/iHMP/diagnose_se_collapse.py`, which classifies
components as collapsed / absorbed / ELBO-blind and writes
`output/se_collapse_diagnosis.csv`.

### Diagnostic output (25 components per group, fresh refits)

| group | stored DE | variance | contribution SD | d_ELBO | d_predictive deviance |
|---|---|---|---|---|---|
| pinned, high DE | 1.000 | 1e-10 | 4.9e-11 | -5.5e-07 | -3.5e-07 |
| pinned, zero DE | 0.000 | 1e-10 | 3.6e-08 | -2.6e-06 | -4.5e-03 |
| unpinned | 0.195 | 0.130 | 0.220 | +3.85 | +14.5 |

The two pinned groups are physically indistinguishable -- same floor
variance, same negligible contribution to predictions, same negligible ELBO
and predictive-deviance change when dropped. Only their *stored* DE differs,
by 1.000 vs 0.000. That is the direct demonstration that DE is noise for a
dead component. Live components behave as expected: dropping one costs 3.85
ELBO and 14.5 predictive deviance units.


## 21. `log_bf` charges a component's own prior penalty against its fit gain (2026-08-26)

Follow-up to section 20, from asking why proline's *live* SE[time_from_max]
scores log_bf=-5.5 while visibly improving the fit.

`calc_metric(metric="BIC")` differences `log_posterior_density`, which is
ELBO + log priors. Dropping a component therefore removes both its fit
contribution and its own prior penalty. For proline these cancel almost
exactly:

```
  ELBO          delta = +1.7812     the component helps the fit
  log priors    delta = -1.7842     removing it removes its prior penalty
  log posterior delta = -0.0030     what log_bf differences
```

The near-exact cancellation is not coincidence: at the optimum of a
penalized objective the marginal likelihood gain and marginal prior cost
balance by construction, so this is systematic rather than a proline quirk.

Measured over 25 live SE components (fresh refits):

| quantity | median | min | max |
|---|---|---|---|
| ELBO delta (actual fit gain) | 3.850 | 0.753 | 23.532 |
| log-posterior delta (what log_bf sees) | 1.372 | -1.128 | 20.472 |

**Only ~29% of a component's fit gain survives into log_bf** (median ratio
0.289). 22 of the 25 carry a negative log_bf despite a positive ELBO gain.
Paper-wide, 137 of the 154 live SE components (89%) have log_bf < 0. An SE
component needs an ELBO gain of roughly 19 to break even at log_bf = 0,
after the ~71% prior absorption and the 2*ln(238)=10.94 parameter penalty.

### Does this invalidate the significance results?

Probably not, and the reasoning matters. The permutation null computes the
observed and the permuted statistics through the identical code path, so a
systematic downward offset appears on both sides and cancels -- that is
precisely what the permutation design buys. What the effect plausibly costs
is **dynamic range**: if log_bf sits near the penalty floor regardless of
effect size, the statistic discriminates poorly and power falls. That is a
power question, not a validity question, and it is what
`examples/simulations/sim_se_power.py` measures.

Do NOT "fix" this by reverting to a likelihood-only comparison without
first re-validating the null. The frozen refit decision (section: the
removed clamp shortcut) exists because a non-refit evaluation produced a
deterministic log_bf and an artificial point mass that broke the null.
Changing which objective is differenced is the same class of change and
needs the same scrutiny.

## 22. The `calc_bic` prior double-count biases the permutation test against live components (2026-08-26)

Follow-up to section 21, prompted by the right question: if proline's
SE[time_from_max] represents something real, is log_bf simply wrong?

### The defect

`calc_bic(loglik, n, k)` returns `k*log(n) - 2*loglik` and documents
`loglik` as "Log-likelihood of observations under model".
`model_classes.calc_metric` calls it as
`calc_bic(loglik=self.log_posterior_density(data), ...)` -- the log
*posterior*, i.e. log-likelihood + log-prior.

BIC's derivation is `log p(D|M) ~= log L(theta_hat) - (k/2) log n`, in which
the `k log n` term IS the Occam factor approximating the prior's
contribution. Passing the log posterior adds `log p(theta_hat)` on top, so
the prior is counted twice. The function is being called against its own
stated contract.

### It does not explain proline's negative sign

Correcting it moves proline's SE[time_from_max] from -5.48 to -3.69. Still
negative, because the component's fit gain is genuinely smaller than the
two-parameter penalty under every accounting:

```
  gain (ELBO)                1.78     penalty 5.47
  gain (predictive log-lik)  4.62     penalty 5.47
```

The 2.06x peak-to-trough effect accounts for 9.24 of 375 total deviance
units -- 2.5%. Visually striking, modest against the noise. The negative
log_bf is a real statement about the data, not an artifact.

### It DOES bias the permutation test, against true positives

The offset was assumed to cancel because observed and permuted statistics
run through the same code path. It does not, because the two sides differ in
exactly the way that matters: the observed component is alive (so dropping
it removes a substantial prior penalty, which the double-count credits back
to the reduced model) while under permutation the component collapses (so
dropping it removes almost nothing).

Measured on proline's SE[time_from_max], observed vs 10 within-subject
permutation draws (7 of 10 permuted components collapsed):

| formula | observed | null median | excess | z |
|---|---|---|---|---|
| log-posterior (current) | -5.48 | -4.805 | **-0.671** | -0.68 |
| ELBO-only (BIC as defined) | -3.69 | **-5.472** | **+1.781** | +0.94 |

**The sign of the excess flips.** Under the shipped formula the component
looks worse than a random shuffle; under BIC as defined it looks better.

The null medians show the mechanism cleanly. Under ELBO-only a dead
component scores exactly -5.472 = -0.5*2*ln(238), the pure parameter
penalty. Under log-posterior it scores -4.805, the double-counted prior
handing back +0.67. The observed, being alive, has a far larger prior
penalty (1.79) credited to its reduced model, which pushes it below the
dead-component null. **The formula systematically penalizes live components
relative to dead ones**, and live components are the ones with real effects.

This is a bias against true positives, so no null-only calibration
simulation would detect it -- Stage 1 could not have caught it.

### Scope and caveats

- One metabolite, 10 permutations. Direction is clear and mechanistically
  explained; the magnitude is not established.
- Fixing it would NOT have made proline significant: z moves from -0.68 to
  +0.94, still short.
- The null SD also changes (0.992 -> 1.895), so this is not a pure location
  shift and p-values will not move by a predictable amount.
- `calc_bic` also feeds `model_search.kernel_test` and
  `model_fitting.kernel_test_reg`; the earlier addendum established neither
  is reached by the real-data pipeline, but both are live exported API.
- Per section 21's warning, changing which objective is differenced is the
  same class of change as the removed clamp shortcut and requires the null
  to be re-validated, not a one-line edit.
- `sim_se_power.py` is running against the CURRENT formula, so its power
  numbers measure the shipped behaviour including this bias. If the fix
  lands, power must be re-measured.

Reproduce: the script is in the session scratchpad as `prior_bias_test.py`;
it refits proline's full and reduced models under observed and permuted
covariates and reports log_bf both ways.

## 23. SE-power simulation: the test is not the bottleneck, fitting is (2026-08-26)

`examples/simulations/sim_se_power.py`, M=150, B0=10/B1=60, 108 min, zero
failed draws. Settles section 20's open question.

### Headline

| stratum | discoveries | false | realized FDR | power |
|---|---|---|---|---|
| squared_exponential[cindex] | 72 | 0 | 0.000 | **72/77 (93.5%)** |
| lin[cindex] | 4 | 1 | 0.250 | 3/77 |

Power by lengthscale was flat -- 100% / 88% / 95% / 91% at ls_rel 0.25 /
0.5 / 1.0 / 2.0 -- so the LogNormal lengthscale prior does **not**
over-suppress at any smoothness. lin[cindex] is correctly near-blind to a
nonlinear effect (3/77), confirming the strata separate what they should.

Component survival mirrored this: 7/73 alive under the null versus 88-100%
alive across the true-positive arms, with fitted lengthscales recovering
the true values (0.295/0.62/0.944/1.68 against 0.25/0.5/1.0/2.0).

### The mechanism: detection is decided at fitting, not at testing

```
                detected=False   detected=True
  SE alive=False       4               0
  SE alive=True        1              72
```

Four of the five misses are components whose variance collapsed to the
floor during fitting; they are undetectable thereafter (p = 0.5 or 1.0).
Conditional on surviving, detection is 72/73 = 98.6%. Seven true-null
components stayed alive and **none** was falsely detected.

So the permutation test is not the limiting factor. The limiting factor is
whether the optimizer keeps the component alive, which is a property of
effect size, not of the significance machinery. This also means sections 21
and 22's log_bf concerns, while real, are not currently costing much power:
93.5% is achieved *with* the prior double-count in place.

### The caveat that constrains what can be claimed

Every simulated effect was stronger than the real component that prompted
this investigation:

```
  simulated amp_c (component contribution SD, log scale):
      min 0.203   median 0.476   max 1.167
  proline's real SE[time_from_max] contribution SD:  0.179
  simulated effects weaker than proline's:  0 of 77
```

Power was already declining at the bottom of the tested range -- 78.6%
(11/14) for amp_c in (0.20, 0.25], 88.9% in (0.25, 0.35], 100% above 0.35 --
and all five misses fall between 0.204 and 0.305. **The simulation
approached the detection boundary but stopped just above where iHMP's actual
candidate effects sit.**

Proline is consistent with being at that boundary: alive (variance 0.102)
but undetected (q=0.37), where the simulation's alive components at amp_c
>= 0.2 were detected 98.6% of the time.

### What the manuscript can and cannot say

**Can say**, and this is a much stronger claim than "we found nothing":
nonlinear associations of magnitude >= 0.35 (log-scale contribution SD)
would have been detected with ~100% power, and >= 0.2 with ~80-90% power,
at realized FDR 0.000; none were found in 564 metabolites for either tested
covariate.

**Cannot say**: that nonlinear effects below ~0.2 are absent. iHMP's ~95%
dead SE components cannot be distinguished between "no effect" and "effect
too weak to survive fitting". Proline's 0.179 sits in exactly that zone.

**Follow-up that would close it**: rerun with amp_c extended down to
0.05-0.25 to map the boundary. That is the regime the real data occupies,
and it is the only remaining gap in the nonlinearity claim.
