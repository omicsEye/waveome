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
- [ ] **B2. Showcase panels into the notebook.** The 3 main + 1 supplemental
  models are currently produced only by `make_showcase_panels.py`; they
  should appear in the notebook so it is the single reproducible narrative.
- [ ] **B3. Reproducibility check (separate agent).** Verify the notebook
  runs end-to-end with NO locally stored output — no pickles, no CSVs, no
  checkpoint. Expect this to expose the untested create-branches.

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
