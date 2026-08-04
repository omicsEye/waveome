# IMPLEMENTATION_PLAN.md — waveome revision code work

Execution handoff for Claude Code. Companion to `waveome_revision_tracker.md` (decisions)
and `waveome_point_by_point_response.md` (reviewer replies). **The tracker is the source of
truth for decisions; this plan is execution detail regenerable from it.** If a decision
changes mid-implementation, update the tracker first, then this plan.

## How to use this document
- It is a **read-only spec**. Maintain your own progress notes elsewhere.
- Do tasks **in order**, respecting the dependency notes. **Stop at each acceptance check
  for maintainer review** before starting the next task.
- **G1 is RESOLVED (2026-08-02)**: permutation vs. empirical-Bayes fallback decided in favor
  of the fallback (T3'); T3 is not being pursued. See G1's entry below.

---

## Frozen decisions (constraints — do not redesign)
1. **Evidence statistic** = refit ΔBIC (`log BF ≈ −½·ΔBIC`, larger = more evidence).
   ΔELBO is an optional later robustness check, not the headline.
2. **Refit required**: the reduced (component-dropped) model is **re-optimized**, not
   frozen at full-model parameters. Warm-start from full-model params.
3. **Magnitude** = marginal (drop-one) **deviance explained**; σ² kept in per-model detail.
4. **Significance** = empirical-null + Benjamini–Hochberg, **stratified per (kernel,
   covariate) pair**. Empirical p = `(1 + #{null ≥ obs}) / (1 + B)`.
5. **Null source**: simulation = known-null components (and as a cross-check target);
   iHMP = subject-level permutation.
6. **Permutation scheme**: single within-unit **circular shift** for every within-unit-
   varying covariate; **across-unit block permutation** for constant-within-unit
   covariates; **target permuted with adjusters held fixed** (roles user-supplied;
   no-roles default = permute each covariate in turn with the rest held fixed).
7. **`var_cutoff`** unified into one parameter, **demoted to a ~1e-8 numerical pre-filter**
   (not the selection rule).
8. **B-layer input warning** = collinearity/role only; **no bundle handling or detection**;
   batch-safe **hard-halt-unless-acknowledgment-argument**, tiered.
9. **Likelihoods** compared = NB vs log-normal vs ZINB (no gamma/hurdle).
10. **Horseshoe scale (τ / `penalization_factor`) — NOT frozen; decide post-G1 (see T7).** Status quo = fixed `1.0` for every metabolite; CV `penalization_search` exists but is **not** invoked by default. **τ does not enter T0/T1** — ΔBIC compares full-vs-reduced at whatever τ the fit used, so τ is a fixed input on both sides. Do not change the default or wire in CV as part of T0–T6.

## Scope / non-goals
- Edit `waveome/` only; never `multioutput/`.
- No manuscript edits; no hard-coded result numbers.
- Library code only; HPC analysis runs are the maintainer's.
- Do not "improve" methodology; flag concerns instead.

## Verified file & symbol map (symbols, not line numbers — lines will drift)
- `waveome/utilities.py` → `calc_feature_importance_components(...)` — the statistic; currently
  drops a component and predicts **without refitting**.
- `waveome/model_classes.py` → `get_feature_importances(self, return_value="log_bf")` (caller),
  `set_penalization_factor(...)` (τ prior), `cut_kernel_components(self, var_cutoff=0.1)`.
- `waveome/regularization.py` → `cut_kernel_components(model, var_cutoff=0.001)`,
  `make_folds(X, unit_col, k_fold=5, random_seed=None)`.
- `waveome/model_search.py` → `GPSearch`, `penalized_optimization(...)`, `outcome_likelihood`
  plumbing. ⚠ `var_cutoff=0.8` here is the unrelated **plotting** cutoff — leave alone.
- `waveome/likelihoods.py` → `ZeroInflatedNegativeBinomial` (already implemented); NB plumbed
  via `outcome_likelihood="negativebinomial"`.
- `examples/simulations/sim_waveome_hpc_run.py` → single-output sim harness.
- `multioutput/benchmarks/simulation/effects.py` → reusable DGP (read-only reference; copy
  generators into the single-output harness rather than importing across the scope boundary).

---

## Ordered tasks

### T0 — Unify `var_cutoff` (independent warm-up)
Merge the two **significance** cutoffs (`model_classes.cut_kernel_components` 0.1 and
`regularization.cut_kernel_components` 0.001) into one documented parameter, default ~1e-8,
described as a numerical pre-filter (not the selection rule). **Do not touch** the 0.8
plotting cutoff in `model_search.py`.
- **Acceptance:** single source for the cutoff; default documented; existing example
  notebooks/scripts still run; `grep` shows no stray 0.1/0.001 *significance* cutoffs.

### T1 — Refit ΔBIC + deviance-explained (depends on T0)
Refactor `calc_feature_importance_components` (and its `get_feature_importances` caller) to
**refit the reduced model** (warm-started from full-model params) and return, from that one
refit: (a) ΔBIC with `log BF ≈ −½·ΔBIC`, sign so larger = more evidence; (b) marginal
deviance explained. Replace the no-refit plug-in predictive log-density. Keep the old
quantity available behind a flag if cheap.
- **Acceptance:**
  - Warm-start vs cold-start refit agree on a handful of components within a stated tolerance
    (guards against initialization artifacts).
  - Tiny synthetic model with one strong + one null component: strong ΔBIC ≫ null ΔBIC;
    deviance-explained is marginal and in a sensible range.
  - Seed 9102; sign convention documented.

### G1 — GATE: combined compute estimate (item 5) — **RESOLVED (2026-08-02)**
Estimate (≈5 covariates × 200 perms × ~32-min full run ≈ 530+ core-hours *before* reduced
refits) came back prohibitive. **Decision: fallback.** T3 (permutation) will not be pursued;
T3' (hardened-EB fallback) is the real-data significance method — see `waveome_revision_
tracker.md` item 5 and `FINDINGS.md` "T2 (continued)" for the calibration work this made
load-bearing rather than secondary.

### T2 — Empirical-null + BH on the simulation (depends on T1; parallel to G1)
Implement empirical p-values `(1 + #{null ≥ obs})/(1 + B)` **per (kernel, covariate)**, then
BH at target q. Build the null from **known-null simulation components** first.
- **Acceptance:** realized FDR ≈ nominal on the known-null simulation across q ∈ {0.01,
  0.05, 0.10}; per-pair stratification vs pooling reproduces the expected gap (pooling
  over-rejects); seed 9102.

### T3 — Permutation null — **NOT PURSUED (G1 resolved to "fallback")**
Subject-level permutation per Frozen-decision 6. Not being implemented — kept here for
reference in case the compute picture changes. Reuse `make_folds`/`random_seed=9102`
conventions. Default (no roles) = permute each covariate in turn, rest held fixed.
- **Acceptance:** **known-null cross-check** — the permutation null matches the ground-truth
  known-null per (kernel, covariate) in simulation. If a stratum diverges (e.g., circular
  shift insufficient for a low-`nᵢ` covariate), **flag it**; do not silently switch schemes.

### T3' — Hardened-EB fallback (ACTIVE — G1 resolved to "fallback")
Efron two-groups local-fdr **hardened**: half-normal SD factor `1/√(1−2/π)`, Storey π₀,
stratified per (kernel, covariate), cumulative-mean → global FDR.
- **Acceptance:** realized FDR ≈ nominal on the known-null simulation (same check as T2).
  ⚠ The Gaussian-null assumption may not hold for horseshoe-shrunk statistics — verify via
  the cross-check before trusting on real data.

### T4 — B-layer input warning (depends on covariate/role API; independent of G1)
Detect strong within-unit collinearity and (optionally) missing role declarations; emit a
tiered, **batch-safe hard-halt** that requires an explicit acknowledgment argument to
proceed. No bundle handling/detection.
- **Acceptance:** halts on a constructed collinear pair unless the acknowledgment arg is
  passed; runs clean otherwise; **no `input()` prompts** (Pegasus-safe).

### T5 — Likelihood selection wiring (depends on T1; lower priority)
Ensure NB / log-normal (Gaussian on `log1p`) / ZINB are selectable for the iHMP comparison;
selection recalibrated per likelihood.
- **Acceptance:** all three run on a small iHMP-like subset and produce per-likelihood
  selections.

### T6 — Scalability instrumentation (depends on T1; lower priority)
Add `psutil` time/memory/iteration logging to the sim sweep and the iHMP run. No standalone
benchmark.
- **Acceptance:** logs emitted with negligible overhead; values feed the R1.M7 reporting.

### T7 — Horseshoe scale (τ) stability sweep (GATED: post-G1; depends on T1 + T2)
**Do not start before G1** — the sweep multiplies fits by the size of the τ grid, so its affordability is decided by the G1 compute estimate.

Preferred design (option (a) in tracker item 9): keep the default `penalization_factor=1.0` as the reported setting, and add the ability to re-run the penalized fit + selection across a grid, e.g. `penalization_factor ∈ {0.5, 1, 10, 100}`, reporting **how stable the selected set is across the grid** (e.g. Jaccard overlap of selected (kernel, covariate) sets vs the default; count of components selected at every τ vs at only one).
- **Implementation note:** `penalization_factor` is already a parameter of `penalized_optimization` (`model_search.py` ~209) and `set_penalization_factor` (`model_classes.py` ~836) sets `tfd.Horseshoe(scale=1.0/penalization_factor)` on every kernel-variance parameter. **No new penalization machinery** — this is a sweep harness + a stability summary, nothing more.
- **Do NOT** enable the per-metabolite CV `penalization_search` path as part of this task (that is option (c), not chosen).
- **Acceptance:**
  - Sweep runs over the grid on a small synthetic/simulation subset with seed 9102 and emits a per-τ selected set + a stability summary.
  - **Rank-invariance sanity check:** within a single metabolite, the *ranking* of components by the ΔBIC statistic should be largely preserved across τ (shrinkage is monotone, so τ should shift level more than order). If the ranking scrambles across τ, **flag it** — that would contradict the reasoning behind option (a) and is worth stopping for.
  - Report the marginal compute cost of the sweep (fits × |grid|) so the maintainer can decide the reported grid size.

---

## Downstream (maintainer-run HPC, enabled by the above — not interactive Claude Code tasks)
Harder-simulation conditions via the ported `effects.py` generators; GPcounts baseline
(pip/GPflow) and attempted LonGP; full iHMP re-run under the finalized rule; permutation
calibration runs; the likelihood comparison and scalability sweeps. Claude Code may prepare
the scripts/configs for these, but should not launch the long runs.
