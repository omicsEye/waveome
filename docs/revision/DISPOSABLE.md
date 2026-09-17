# Disposable artifacts — what can be deleted once the revision ships

Living list. Nothing here is deleted; this records what is safe to remove
and why, so the decision does not have to be reconstructed later.

`examples/iHMP/output/` is **2.3 GB**, and ~1.9 GB of that is superseded
model pickles rather than results.

---

## KEEP — the reported analysis depends on these

| artifact | size | why |
|---|---|---|
| `fit_penalized_models_revision_full_scipy_ls_prior_no_prune_clamp_removed.pkl` | 172 MB | the reported fit; 13 references across scripts and the notebook |
| `ihmp_permutation_draws.csv` | 5.4 MB | the checkpoint; regenerating costs ~23 h |
| `ihmp_permutation_significance.csv` | 0.3 MB | reported p/q at B0=10, B1=100 |
| `ihmp_between_subject.csv` | 0.1 MB | between-subject complement; regenerating costs ~15 min via the notebook |
| `all_component_results_with_significance.csv` | 0.6 MB | the component table (regenerate when the notebook runs) |
| `showcase_*`, `supp_showcase_*`, `permutation_null_explained*` | 11 MB | the manuscript figures + captions |
| `*_runtime_stats.json` | <1 MB | R1.M7 evidence — NOT disposable until a `..._clamp_removed` one exists; the present three describe earlier fits (TODO B3.4) |

## ARCHIVED — evidence behind FINDINGS, delete once the paper is accepted

`examples/iHMP/output/archive_oldbic_2026-09-01/` (see its README)

- old-BIC permutation draws, significance CSVs, component tables, and figures
  — the evidence for FINDINGS 20-22 (the -4.8 / -1.0 mass points, 95%-dead SE
  components, and the prior double-count)
- `fit_penalized_models_OLDBIC.pkl` (172 MB) — the pre-fix pickle
- `ihmp_permutation_draws_B120tfm.csv` — the 120-draw time_from_max checkpoint,
  kept when the analysis standardised at B=100
- `superseded_diagnostic_figures/` (14 files) — hardened-EB-era null shapes,
  log_bf histograms, QQ and sigma-sensitivity checks. Several depict methods
  that were rejected; all predate the calc_bic correction, so any log_bf in
  them is on the superseded statistic.

## DELETABLE NOW — superseded fits, zero references

Checked by grepping every script, notebook and doc outside `output/`.

| artifact | size | refs |
|---|---|---|
| `ihmp_waveome_output.pickle` | 421 MB | 1 (a stale path in an old notebook cell) |
| `fit_penalized_models.pkl` | 145 MB | 0 |
| `fit_penalized_models_v1.0.pkl` | 145 MB | 0 |
| `fit_penalized_models_v1.0_updated.pkl` | 145 MB | 0 |
| `fit_penalized_models_v2.pkl` | 145 MB | 0 |
| `fit_penalized_models_adam_v1.0.pkl` | 145 MB | 0 |
| `fit_penalized_models_adam_v1.1.pkl` | 145 MB | 0 |
| `fit_penalized_models_adam_gradient.pkl` | 154 MB | 0 |
| `fit_penalized_models_adam_gradient_previous.pkl` | 154 MB | 0 |
| `fit_penalized_models_revision_full_scipy_no_ls_prior.pkl` | 156 MB | 0 |
| `draft_figs/` | ~2 MB | manuscript figure candidates for inspection; delete once Fig. 6/7 are chosen |

**~1.75 GB.** All predate this revision or are intermediate fits from it
(pre-lengthscale-prior, pre-no-prune). None is referenced by any live code
path. Kept only because deleting a fit is irreversible without a ~98 min
re-run.

Two with a single reference each, so check before removing:

- `fit_penalized_models_revision_full_scipy_ls_prior.pkl` (156 MB) — the
  pre-no-prune fit, referenced in a comparison script
- `fit_penalized_models_revision_full_scipy_ls_prior_no_prune.pkl` (172 MB) —
  the pre-clamp-removal fit, referenced in a comparison script

## TEMPORARY — outside the repo, auto-cleaned

Session scratchpad at
`/private/tmp/claude-501/.../scratchpad/`: run logs
(`ihmp_perm_*.log`, `stage2_shipped.log`, `se_power.log`, `figs_*.log`),
executed notebook copies, and one-off diagnostics
(`prior_bias_test.py`, `dead_stability.py`, `absorb_test.py`). Nothing
depends on these; they disappear with the session.

Also transient in-repo:

- `examples/iHMP/output/*.log` — run logs (`ihmp_permutation.log`,
  `ihmp_permutation_killed_run.log`, `tfm_topup.log`, `between_subject.log`)
- `examples/iHMP/output/within_perm_{smoke,mix}_*.csv` — early prototype runs
- `examples/iHMP/output/.DS_Store`
