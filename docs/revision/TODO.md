# Open work — significance methodology

Ordered by dependency. Items that change results come first, so downstream
artifacts are regenerated once rather than repeatedly.

Status: `[ ]` open · `[~]` in progress · `[x]` done

---

## A. Items that CHANGE RESULTS (do first)

- [ ] **A1. `C18n_QI43` SE[hbi] false positive.** A dead component (variance
  on VARIANCE_FLOOR) scoring q=0.032. Survives at B=100 because the failure
  is structural, not sampling: its observed statistic sits above its
  jitter-scale null via leakage from that metabolite's `lin:hbi`
  (log_bf 8.38). FINDINGS 26, 28.
  - Inflates hbi from 167 to 168 components — the sole cause of the
    notebook's component/metabolite gap.
  - Fix: gate degeneracy on fitted **variance**, not null SD, in
    `calc_permutation_pvalues`. Frozen component → needs sign-off.
  - **Changes:** significance CSV, all figures, notebook cells 12/15/22.

## B. Downstream of A — regenerate once A is settled

- [ ] **B1. Regenerate** significance CSV, 5 figures, notebook.
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
