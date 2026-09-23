# Manuscript changes required by the revised significance methodology

Source: `waveome_manuscript_submitted/sn-article.tex` (submitted version) checked
against `examples/iHMP/output/ihmp_permutation_significance.csv` (within-subject
permutation, B0=10 / B1=100, BH within stratum, q<=0.10) and
`examples/iHMP/output/ihmp_between_subject.csv`.

Status: `[ ]` open · `[~]` in progress · `[x]` done

**Every count and log Bayes factor in the iHMP sections changes.** The submitted
results used the variance>1e-4 rule and the pre-correction `calc_bic`, which
double-counted the lengthscale prior. Both are gone.

---

## A. Claims that are now UNSUPPORTED — highest priority

- [x] **M1. The SCFA claim fails — DONE (sn-article-revised.tex).** Five sites, not the three originally listed: abstract, Results L155 (class list), Discussion L221 (class list), L222 (literature support), L225 (mechanism clause). The Results copy of the six-class list was missed on the first pass and found by re-grepping.

  Rather than delete the SCFA finding, the Discussion now states it as a null result and explains it: the established SCFA depletion is a case/control contrast against healthy individuals, while this analysis asks whether a metabolite tracks severity *within* a Crohn's cohort. Numbers quoted: butyrate/propionate/valerate-isovalerate q=1.00, caproate q=0.27.

  Three citations are now unused (`kaczmarczyk_altered_2022`, `parada_venegas_short_2019`, `xu_characterization_2022`) — harmless to LaTeX, but drop them from the .bib if the journal objects.

  ORIGINAL FINDING: **The SCFA claim fails.** Abstract: "we recover well-established
  biomarkers, such as short-chain fatty acids, secondary bile acids, and
  specific lipid species". Discussion repeats it with citations. Under the
  permutation criterion **no canonical SCFA is significant for HBI**:
  butyrate q=1.00, propionate q=1.00, valerate/isovalerate q=1.00,
  caproate q=0.27. Bile acids and lipids DO survive (glycodeoxycholate
  log_bf=10.79 q=0.0018, glycochenodeoxycholate 10.50 q=0.0003, plus PC/SM/LPC
  species), so the sentence needs the SCFA term removed, not deleted.
  Also affects Discussion para 2 ("SCFAs and bile acids are consistently
  depleted...") and para 3 ("reduced SCFAs compromise gut barrier integrity").

- [ ] **M2. Oxalate is no longer significant — and it has its own figure.**
  Submitted Table 2 leads with oxalate at log_bf=47.7. Now log_bf=-2.74,
  q=1.00 on *all four* strata: it sits at the exact dead-component parameter
  penalty. Figure `oxalate_parts_output.png` (Fig. 7) and its caption must be
  replaced, and the Discussion sentence citing oxalate/CD urolithiasis
  (6 references) removed or requalified.

- [ ] **M3. The lithocholate figure's stated mechanism is not supported.**
  Caption of `lithocholate_hbi.png` (Fig. 6) claims "a common squared
  exponential HBI kernel where higher HBI is associated with lower
  lithocholate". Lithocholate's `SE:hbi` is now q=1.00. One lithocholate
  feature is significant on `lin:hbi` (q=0.0007) but at **log_bf=-1.01**, so
  it cannot carry a "strong association" narrative either. Replace the figure
  or rewrite to a linear claim on the surviving feature.

- [ ] **M4. Table 2 (temporal associations) loses 8 of 9 rows.**
  Current time_from_max hits, complete list (6 metabolites, not 9):

  | Metabolite | Compound | Kernel | log_bf | q |
  |---|---|---|---|---|
  | C56:6 TAG | C8p_QI201 | lin | 3.04 | 0.0080 |
  | 3-hydroxymethylglutarate | HILn_QI9 | lin | 2.17 | 0.0080 |
  | serine | HILn_QI110 | lin | 1.35 | 0.0080 |
  | NH4_C56:2 TAG | C8p_QI208 | lin | 0.41 | 0.0080 |
  | nervonic acid | C18n_QI89 | SE | -1.07 | 0.0159 |
  | NH4_C52:6 TAG | C8p_QI171 | SE | -2.80 | 0.0159 |

  Of the submitted nine, only **C56:6 TAG** survives under the same compound.
  Dropped: oxalate (q=1.00), betaine (q=1.00 temporal — still significant for
  HBI, log_bf=11.49, so it moves sections), taurolithocholate (q=0.485),
  4-methylcatechol (q=1.00), C42:0 TAG, C52:5 TAG, C20:5 CE.
  `NH4_C52:6 TAG` is a different adduct/feature from the submitted
  "C52:6 TAG" — confirm before treating it as the same finding.
  The "Relationship" column (e.g. "Elevated pre and post-max") must be
  re-read off the new fits, not carried over.

## B. Counts and numbers

- [ ] **M5. "Seventy-two metabolites showed significant associations with HBI"
  -> 167.** Metabolite-level, q<=0.10. Component-level the strata are
  `lin:hbi` 165, `SE:hbi` 3, of 564 tested each.
- [ ] **M6. "Nine metabolites exhibited significant temporal associations"
  -> 6.** See M4.
- [ ] **M7. Every log Bayes factor in the text, Table 2, and the heatmap
  changes** — the `calc_bic` correction removed a double-counted prior term.
  Do not carry any submitted number forward.
- [ ] **M8. "These metabolites clustered into six functional classes"** —
  re-derive from the new 167. The class list is currently asserted from the
  old 72 and at least one class (SCFAs) no longer has a member (M1).

## B2. BLOCKING BUG — component panel directions

- [~] **M24. FIXED in `waveome/utilities.py`; figures still need regenerating.**
  `plot_parts` component curves were not an additive decomposition
  (FINDINGS 29).** `individual_kernel_predictions(marginal=True)` isolates a
  component by swapping the model's kernel for that sub-kernel, but the SVGP
  is whitened against the FULL `K_zz`, so the fitted `q_mu` is reinterpreted
  against the wrong matrix. The components do not sum to the full model
  (residual spread 2.26 vs 3.8e-10 for the correct shared-alpha
  decomposition), and **the sign can flip**: 3-hydroxymethylglutarate's
  `lin[hbi]` plots as FALLING while the model actually RISES with HBI.

  **Significance is unaffected** (log_bf/q/DE come from drop-one refits, not
  from this function), and **conditional figures are unaffected**
  (`plot_marginal` uses the full model). But **every showcase panel shape and
  direction is suspect**, including:
  - bilirubin's SE[hbi] sigmoid
  - nervonic acid's SE[tfm] peak at max severity
  - the "Relationship" column that would replace Table 2 (M4)

  **Fixed 2026-09-12** (FINDINGS 29, `tests/test_component_decomposition.py`);
  additivity residual on the real model is 4.3e-14. **Still to do: regenerate
  every `plot_parts` figure and re-read the directions**, including the
  draft_figs candidates, before any direction is written into the manuscript.

## C. Figures to regenerate

- [ ] **M9. `ihmp_hbi_heatmap.png` (Fig. 5), "top twenty by log Bayes
  factor".** Entirely new membership. New top 20: sorbitol (27.83),
  lactate (21.07), docosahexaenoate (18.90), succinate (13.61), malate
  (13.22), eicosadienoate (12.75), C32:0 PC (12.63), C16 carnitine (11.80),
  betaine (11.49), arachidonate (11.43), glycodeoxycholate (10.79),
  glycochenodeoxycholate (10.50), 1-methylguanosine (10.44), choline (10.32),
  C18:1 LPC plasmalogen (10.17), C16:0 SM (9.74), urate (9.57), C18:1 LPC
  (9.45), C34:1 PC plasmalogen (9.40), glutamine (9.22).
- [~] **M10. Replace Fig. 6 (lithocholate) and Fig. 7 (oxalate).** Draft
  candidates rendered to `output/draft_figs/` (disposable). Component
  liveness verified against actual kernel variances, not `deviance_explained`
  — DE is unreliable (butyrate shows 13 components >5% DE with 0 alive).

  **CHOSEN (2026-09-13), regenerated under the FINDINGS 29 fix:**
  - **Fig. 6 — sorbitol (HILn_QI112)**, HBI, conditional on individuals.
    `output/conditional_hbi_HILn_QI112.{png,pdf}`
  - **Fig. 7 — nervonic acid (C18n_QI89)**, time_from_max, additive, SE.
    `output/showcase_time_from_max_C18n_QI89_SE.{png,pdf}`
  - **Supplemental — bilirubin (HILp_QI19549)**, HBI, additive, SE.
    `output/supp_showcase_hbi_HILp_QI19549_SE.{png,pdf}`

  Nervonic acid is C24:1 and `C24:1 SM` is independently significant for HBI
  (log_bf 7.62), alongside C24:0/C22:0/C22:1/C16:0 SM — the figure
  illustrates a sphingolipid block rather than a singleton. Side-verification
  (not for the manuscript): 98.4% of posterior draws put an interior peak in
  the SE[tfm] curve, but the peak's 95% interval is -148 to +25 days, so the
  shape is well supported and the timing is not. Do not state a lag.

  **Superseded reasoning, kept for the record:**
  - **Fig. 6 — sorbitol (HILn_QI112), HBI, conditional on individuals.**
    Alive: cat[participant_id] 0.377, cat[race] 0.119, lin[hbi] 0.106.
    log_bf 27.83 (highest of all 167 hits), q=0.0010. Per-participant
    offsets are clearly separated and the HBI rise is legible.
  - **Fig. 7 — nervonic acid (C18n_QI89), time_from_max, additive.**
    Alive: SE[tfm] 0.111, SE[hbi] 0.050, cat[sex] 0.026, lin[hbi] 0.017,
    lin[study_days] 0.016. q=0.0159, DE 41.5%. The SE[tfm] panel peaks
    exactly at the max-severity point — the dynamic story the oxalate
    figure was telling. **Cost: log_bf = -1.07** (see M20).

  **Rejected after inspection:**
  - C56:6 TAG conditional — outliers compress the y-axis, one of four
    participants is flat. Visually much weaker than sorbitol.
  - 3-hydroxymethylglutarate (HILn_QI9) additive is the numerically
    strongest temporal option (68.5% deviance explained, TWO significant
    components: lin[hbi] q=0.0003 and lin[tfm] q=0.008, both positive
    log_bf) but is entirely linear — no SE anywhere in the figure set.
    **Use this instead of nervonic acid if a negative log_bf in a main
    figure is unacceptable.**
  - Bilirubin additive (SE[hbi] 0.412 dominant, DE 56.7%, q=0.0092,
    log_bf +0.25) is a good figure and the only SE claim with positive
    log_bf. Suggest keeping it as a supplemental/third panel so the
    nonlinearity claim is carried by a positive log_bf somewhere.
- [ ] **M11. Confirm the simulation figures need no change.**
  `sens_spes_sim_cross.png` and `sim_kl_divergence.png` predate the `calc_bic`
  correction, which touches `calc_metric` and therefore the *search* variant.
  **Unverified — flag for the maintainer**, this is an HPC re-run, not
  something to settle here.

## D. Methods text

- [ ] **M12. Rewrite "Significance criterion and reproducibility".** The
  submitted rule — "a metabolite was deemed to show a significant association
  if the corresponding kernel variance parameter remained above 1e-4" — is
  exactly what the reviewer objected to and is no longer what the code does.
  Replacement: drop-one refit dBIC -> log_bf = -0.5*dBIC, warm-started from the
  full model; within-subject free-shuffle permutation null (B0=10 screen,
  B1=100 derived as (m/n_live)/q_target); conditional quantile regression of
  pooled centred draws on each test's null SD; BH within stratum at q<=0.10;
  seed 9102.
- [ ] **M13. Add the between-subject complement.** New method, no text yet.
  Within-subject permutation preserves each unit's covariate values, so
  between-unit association is untestable by it — and HBI carries 41% of its
  variance between units, time_from_max 47%. `between_subject_significance`
  covers it: refit without the covariate, per-unit mean covariate vs mean
  Pearson residual, Spearman, permute unit means (20,000 draws), BH.
  **Result: 0/564 for both covariates** — a null result that needs reporting,
  since it bounds what the cross-sectional HBI claims can mean.
- [ ] **M14. State the multiplicity structure.** 564 metabolites x 2 kernels x
  2 covariates; BH applied within each of the four strata. The submitted text
  has no multiplicity control at all.

## E. Showcase figures — what the results now support

The notebook (cell 21) produces four panels. Their status against the above:

- [ ] **M15.** `proline` (HILp_QI578) — linear HBI, richest decomposition of
  the 165 hits (6 components >5% DE). Uncontested.
- [ ] **M16.** `serine` (HILn_QI110) — linear time_from_max, richest of the 4
  (4 components vs 1). Uncontested, and it is one of only 6 temporal hits.
- [ ] **M17.** `bilirubin` (HILp_QI19549) — the nonlinearity claim:
  `lin[hbi]` q=1.00 while `SE[hbi]` q=0.0092, i.e. a linear model misses it
  entirely. **This is the natural replacement for the lithocholate figure
  (M3)**, which made an SE claim the data no longer supports.
- [ ] **M18.** `metronidazole` (HILp_QI2850) — supplemental positive control,
  SE:hbi log_bf=4.90 q=0.0092. Confounding by indication (antibiotic for
  active Crohn's), so it validates the method rather than the biology.
- [ ] **M19. Consider a temporal showcase to replace oxalate (M2).** Serine
  (M16) already covers linear time_from_max. If a temporal *nonlinear* panel
  is wanted, the only candidates are nervonic acid (log_bf=-1.07) and
  NH4_C52:6 TAG (-2.80), both with negative log_bf — see M20.

## F. Presentation problems to decide on

- [ ] **M20. 29 of 168 significant HBI components have NEGATIVE log_bf**
  (min -5.47, median +2.95). A component can clear BH while dropping it
  *improves* BIC, because significance is judged against the permutation null,
  not against zero. The heatmap colors by log_bf, so these render as evidence
  *against* an association they are reported as supporting. Options: report
  q and log_bf side by side; restrict figures to log_bf>0; or state the
  distinction explicitly. **Methodology is frozen — this is a reporting
  decision, not a criterion change.**
- [ ] **M21. Adrenate (C18n_QI43) is a known artifact.** It reaches SE:hbi
  significance on a floor-variance (dead) component. It is separately
  significant via lin:hbi, so the metabolite-level count of 167 is unaffected,
  but **do not report "3 nonlinear HBI associations" — it is 2**
  (metronidazole, bilirubin). See TODO A1.
- [ ] **M22. Ties at the resolution floor.** Many HBI hits share the minimum
  attainable q (q=0.0003 at B=100), so they cannot be ranked against one
  another. Report those q values as bounds, and do not describe any of them as
  "the most significant".
- [ ] **M23. Borderline membership is unstable.** Going B=60 -> B=100 lost 6
  hits and gained 4; the count is stable, the membership is not. Manuscript
  claims should concern the population of hits, not named borderline
  metabolites. (FINDINGS 27-28.)

  Recorded in `waveome_revision_tracker.md` under R1.M5/R2.5 (TODO C2, done).
  Concretely, the text may say "167 metabolites associate with HBI" but must
  not name a metabolite whose q sits within a few percent of 0.10 as though
  the cutoff were sharp. The chosen figures are safe: sorbitol log_bf 27.83,
  and neither nervonic acid nor bilirubin is near the boundary.
