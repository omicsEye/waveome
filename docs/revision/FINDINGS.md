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
