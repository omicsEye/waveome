# Open work — significance methodology

Ordered by dependency. Items that change results come first, so downstream
artifacts are regenerated once rather than repeatedly.

Status: `[ ]` open · `[~]` in progress · `[x]` done

---

## A. Items that CHANGE RESULTS

- [x] **A1. `C18n_QI43` SE[hbi] false positive — RESOLVED, no code change.**
  The reporting unit is the METABOLITE, not the component: the research
  question is which metabolite intensities associate with disease activity,
  not which functional form carries the association.

  Verified that this framing is safe rather than merely convenient: **no
  metabolite reaches significance solely through a dead component.**
  C18n_QI43 is the only hit on a floor-variance kernel, and it is separately
  significant via `lin:hbi` at log_bf 8.38, q=0.0003. So the metabolite-level
  answer is **167 for hbi and 6 for time_from_max**, artifact or not.

  `calc_permutation_pvalues` is therefore left alone -- it is frozen, it
  carries Stage 1 / Stage 2 / sim_se_power validation, and changing it would
  buy nothing at the level results are reported.

  **The one live constraint:** do not report the SE:hbi stratum count of 3
  as "3 nonlinear associations" -- one is the artifact, so it is 2. The
  nonlinearity showcase (bilirubin, metronidazole) is unaffected; both are
  alive. Component-level counts need the dead-kernel check; metabolite-level
  counts do not.

  **Nothing downstream changes.** No re-run, no regeneration.

## B. Notebook work (unblocked -- A1 needs no re-run)

- [x] ~~B1. Regenerate~~ — not needed; A1 changes no results.
- [x] **B2. Showcase panels into the notebook — DONE.** The 3 main + 1 supplemental
  models are currently produced only by `make_showcase_panels.py`; they
  should appear in the notebook so it is the single reproducible narrative.
- [~] **B3. Reproducibility check (separate agent) — IN PROGRESS.** Verify the notebook
  runs end-to-end with NO locally stored output — no pickles, no CSVs, no
  checkpoint. Expect this to expose the untested create-branches. Findings,
  in severity order:

  - [x] **B3.1 `statsmodels` was an undeclared dependency.** Added to
    `pyproject.toml` (`97c6166`).
  - [x] **B3.2 Between-subject block vanished silently on a clean checkout.**
    `ihmp_between_subject.csv` is untracked and was produced by a separate
    script, so the notebook printed one advisory line and dropped the entire
    block. Cell 12 now calls `gps.between_subject_significance` under the
    same load-or-create pattern as the fit. Verified the library method
    reproduces the retired script over all 1128 outcome x covariate pairs
    (max abs diff 1.1e-16 on p and q; `0/564` for both covariates), then
    retired `run_between_subject_test.py` so one implementation remains.
  - [x] **B3.3 Selection rule printed but not applied — RESOLVED by removing
    the claim.** The rule was never applied to the SE panels, and once the
    figures were chosen on biological coherence and model structure it became
    simply false. Cell 21, markdown cell 20 and the caption file now state the
    per-figure criterion instead of asserting a single mechanical rule, and
    print how many significant metabolites each figure was chosen from.
  - [x] **B3.4 Provenance — DONE.**
    - [x] **Duplicate component table removed.** Cells 18 and 19 wrote the
      same dataframe to `all_component_results.csv` and
      `all_component_results_with_significance.csv`, so the two files were
      byte-identical (md5 `861f45aa…`) and neither name told you which was
      current. Cell 18 no longer writes; the `_with_significance` name is
      kept because `diagnose_se_collapse.py` reads that path.
    - [x] **R1.M7 operating point — RESOLVED 2026-09-16.** The maintainer
      re-ran the fit; `output/ihmp_penalized_fit_runtime_stats.json` now
      measures the pickle every reported result uses.

      ```
      wall_clock_min      89.65      sec_per_metabolite   9.538
      peak_memory_gb      20.68      convergence_rate     0.982
      mean_n_iterations  294.9       median_n_iterations  280.0
      n_metabolites        564       n_models_with_stats   564
      ```

      **Report peak memory WITH the hardware.** Iterations and convergence
      match the pre-fix proxy almost exactly (294.9 vs 294.7; 0.982 vs
      0.982), so the fit behaved identically -- but peak memory is 20.68 GB
      against the proxy's 8.37 GB. That difference tracks core count and
      parallel worker layout, not the method, and will be read as a method
      property if reported bare.

      **Re-fit reproduced the artifact.** 560 of 564 models bitwise
      identical. Of the 4 that differ, 3 differ only in a
      `categorical[participant_id]` variance at ~1e-4 relative and have no
      live tested component (every log_bf pinned at the exact parameter
      penalty, q=1.00), so their statistics cannot move. The fourth,
      `HILp_TF42`, IS a reported hit (`lin:hbi`, q=0.0086) and its
      `lin[hbi]` variance moved 1.79% relative -- but its statistic is
      unchanged to four decimals on both pickles (log_bf 4.7000,
      DE 50.40%), because the drop-one refit lands in the same place.
      The backup `ihmp_penalized_fit_PREFIT_BACKUP.pkl` can be deleted once
      this is considered settled.

      **New, unrelated, small:** a fresh recomputation of `HILp_TF42`'s
      statistic gives log_bf 4.7000 where
      `ihmp_permutation_significance.csv` stores 4.68. The gap is present on
      BOTH pickles, so it predates the re-fit -- the stored CSV and a fresh
      recompute disagree slightly somewhere. Not urgent, but worth resolving
      before the numbers are final.

      Historical context for the above:
      The three earlier
      `*_runtime_stats.json` files belonged to earlier configurations, and
      the pickle behind every reported result had none because it predated
      the instrumentation in cell 11's fit branch.

      | stats file | prune | convergence | wall clock | sec/metabolite | peak GB |
      |---|---|---|---|---|---|
      | `..._full_scipy` | True | 0.266 | 80.0 min | 8.51 | 7.55 |
      | `..._ls_prior` | True | 0.264 | 78.5 min | 8.35 | 8.04 |
      | `..._ls_prior_no_prune` | False | **0.982** | 97.9 min | 10.42 | 8.37 |
      | `..._no_prune_clamp_removed` | False | — | **MISSING** | — | — |

      `..._ls_prior_no_prune` is the closest proxy — same configuration
      except the clamp removal — but the clamp removal was a numerical fix
      to the optimiser (FINDINGS 1-2), so convergence and iteration counts
      are exactly what it could move. **Do not quote the proxy as the
      operating point.** Cell 11 now prints a NOTE on load when the stats
      file is absent, so the gap surfaces on every run instead of in an
      audit.

      **Artifacts renamed** for outside reviewers -- the old name encoded the
      development history rather than what the file is:

      | | |
      |---|---|
      | model | `output/ihmp_penalized_fit.pkl` |
      | stats (to be produced) | `output/ihmp_penalized_fit_runtime_stats.json` |
      | pre-refit backup | `output/ihmp_penalized_fit_PREFIT_BACKUP.pkl` |
      | proxy stats, earlier config | `output/ihmp_penalized_fit_PROXY_no_clamp_fix_runtime_stats.json` |

      All 14 code and doc references updated; two of them
      (`run_ihmp_permutation.py`, `refresh_feature_importances.py`) split the
      name across concatenated string literals and needed patching by hand.

      **Re-fit protocol.** Cell 11 writes to the same path, and the 23 h
      permutation results describe the CURRENT pickle, so a non-reproducing
      re-fit would strand them:
      1. backup already taken (`..._PREFIT_BACKUP.pkl`)
      2. `rm output/ihmp_penalized_fit.pkl`, run cell 11 (~100 min)
      3. verify per-model kernel variances across all 564 models, new vs
         backup
      4. match -> keep; differ -> restore the backup, and the new stats are
         unusable for the same reason the old ones are

      Config was checked against the pickle's own stored `run_parameters`
      before renaming: seed 9102, `num_restart=3`, `prune_components=False`,
      scipy, `[SquaredExponential, Lin]`, all interaction flags False -- every
      value matches what cell 11 would call, so this re-runs the same fit.
  - [x] **B3.5 Cosmetic — DONE.** Four items, one of which was not cosmetic:
    - Notebook header said `waveome v0.1.0`; pyproject says `0.2.0`. There is
      no `__version__` anywhere in the package, so nothing else was stale.
    - `../iHMP/data/...` replaced with `data/...` in the notebook, and
      `ihmp_waveome_hpc_run.py` now resolves `DATA` from `__file__` instead.
      The old relative path only worked when launched from a SIBLING of
      `examples/iHMP`, so it broke silently depending on submit directory.
    - Five `FINDINGS...` references given the `docs/revision/` prefix so they
      locate the file.
    - **`id_list` collision (a real bug, not cosmetic).** Cell 16 rebound
      `id_list`, the name cells 8-9 use for the sampled participants in the
      trajectory figure, to `[19, 13, 27, 17]`. Re-running cell 9 after cell
      16 silently plotted the wrong participants. Cell 16's is now
      `marginal_example_ids`.

## C. Documentation only — no result changes, can run in parallel

- [ ] **C1. Tracker row R1.M5/R2.5** still cites the superseded Stage 2
  evidence (0.010/0.029/0.029 from the *binned* method). FINDINGS 25 has the
  replacement from the shipped construction.
- [ ] **C2. Record the borderline-stability caveat.** 6 hits lost, 4 gained
  going B=60 -> B=100; the count is stable, individual membership is not.
  Manuscript claims should concern the population of hits, not specific
  borderline metabolites. FINDINGS 27, 28.

---

## Known-but-accepted (not scheduled)

- `deviance_explained` is meaningless for dead components (FINDINGS 20).
  Fenced by documentation, not by code.
- B0=10 misses ~2% of live components; the miss is silent and permanent.
- Adaptive/sequential stopping (Gandy 2009, MMCTest) would target the
  borderline instability directly. Deemed overkill for this revision.
