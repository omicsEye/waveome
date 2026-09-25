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

- [x] **M2. DONE.** Oxalate is no longer significant — and it has its own figure.**
  Submitted Table 2 leads with oxalate at log_bf=47.7. Now log_bf=-2.74,
  q=1.00 on *all four* strata: it sits at the exact dead-component parameter
  penalty. Figure `oxalate_parts_output.png` (Fig. 7) and its caption must be
  replaced, and the Discussion sentence citing oxalate/CD urolithiasis
  (6 references) removed or requalified.

- [x] **M3. DONE.** The lithocholate figure's stated mechanism is not supported.
  Caption of `lithocholate_hbi.png` (Fig. 6) claims "a common squared
  exponential HBI kernel where higher HBI is associated with lower
  lithocholate". Lithocholate's `SE:hbi` is now q=1.00. One lithocholate
  feature is significant on `lin:hbi` (q=0.0007) but at **log_bf=-1.01**, so
  it cannot carry a "strong association" narrative either. Replace the figure
  or rewrite to a linear claim on the surviving feature.

- [x] **M4. DONE.** Table 2 (temporal associations) rebuilt, 9 rows -> 6.
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

- [x] **M5. DONE.** "Seventy-two metabolites showed significant associations with HBI"
  -> 167.** Metabolite-level, q<=0.10. Component-level the strata are
  `lin:hbi` 165, `SE:hbi` 3, of 564 tested each.
- [x] **M6. DONE.** "Nine metabolites exhibited significant temporal associations"
  -> 6.** See M4.
- [x] **M7. DONE by verification — no stale values remain in the text.** Every
  numeric claim in the iHMP Results and Discussion was re-checked against
  `ihmp_permutation_significance.csv` on 2026-09-24 and matches: sorbitol
  27.827/q=0.0010, C24:1 SM 7.620/q=0.0003, betaine 11.491/q=0.0003, nervonic
  acid -1.066/q=0.0159, butyrate q=1.0000, caproate best q=0.2714, and all six
  Table 2 rows. The M1-M6 rewrites had already replaced every value carried
  over from the submitted version; nothing was left behind. The heatmap figure
  itself is still stale (M9).

  ORIGINAL: **Every log Bayes factor in the text, Table 2, and the heatmap
  changes** — the `calc_bic` correction removed a double-counted prior term.
  Do not carry any submitted number forward.
- [x] **M8. CLOSED by author decision — keep it general.**
  The class list is descriptive orientation for the reader, not a result the
  paper argues, so it stays qualitative rather than being re-derived as a
  curated six-class breakdown. Original finding: "These metabolites clustered
  into six functional classes" —
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

- [x] **M9. DONE — figure TYPE changed, see the rationale block below.**
  ORIGINAL: `ihmp_hbi_heatmap.png` (Fig. 5), "top twenty by log Bayes
  factor".** Entirely new membership. New top 20: sorbitol (27.83),
  lactate (21.07), docosahexaenoate (18.90), succinate (13.61), malate
  (13.22), eicosadienoate (12.75), C32:0 PC (12.63), C16 carnitine (11.80),
  betaine (11.49), arachidonate (11.43), glycodeoxycholate (10.79),
  glycochenodeoxycholate (10.50), 1-methylguanosine (10.44), choline (10.32),
  C18:1 LPC plasmalogen (10.17), C16:0 SM (9.74), urate (9.57), C18:1 LPC
  (9.45), C34:1 PC plasmalogen (9.40), glutamine (9.22).
- [x] **M10. DONE — both figures are now in sn-article-revised.tex.** Draft
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

- [x] **M12. DONE.** Rewrote "Significance criterion and reproducibility". The
  submitted rule — "a metabolite was deemed to show a significant association
  if the corresponding kernel variance parameter remained above 1e-4" — is
  exactly what the reviewer objected to and is no longer what the code does.
  Replacement: drop-one refit dBIC -> log_bf = -0.5*dBIC, warm-started from the
  full model; within-subject free-shuffle permutation null (B0=10 screen,
  B1=100 derived as (m/n_live)/q_target); conditional quantile regression of
  pooled centred draws on each test's null SD; BH within stratum at q<=0.10;
  seed 9102.
- [x] **M13. DONE.** Added the between-subject complement. New method, no text yet.
  Within-subject permutation preserves each unit's covariate values, so
  between-unit association is untestable by it — and HBI carries 41% of its
  variance between units, time_from_max 47%. `between_subject_significance`
  covers it: refit without the covariate, per-unit mean covariate vs mean
  Pearson residual, Spearman, permute unit means (20,000 draws), BH.
  **Result: 0/564 for both covariates** — a null result that needs reporting,
  since it bounds what the cross-sectional HBI claims can mean.
- [x] **M14. DONE.** Multiplicity structure stated. 564 metabolites x 2 kernels x
  2 covariates; BH applied within each of the four strata. The submitted text
  has no multiplicity control at all.

## E. Showcase figures — what the results now support

The notebook (cell 21) produces four panels. Their status against the above:

- [x] **M15. SUPERSEDED — proline is not in the manuscript.** It was the
  candidate when selection was by "richest decomposition"; the HBI figure is
  now sorbitol, chosen because its model is an individual offset plus a common
  linear HBI kernel (the structure the figure needs) and it carries the
  largest log BF of all 167 hits. Verified: 0 mentions of proline in
  sn-article-revised.tex. ORIGINAL: `proline` (HILp_QI578) — linear HBI, richest decomposition of
  the 165 hits (6 components >5% DE). Uncontested.
- [x] **M16. SUPERSEDED as a FIGURE; serine remains a RESULT.** The temporal
  figure is nervonic acid, which carries the nonlinear claim and the
  sphingolipid coherence. Serine is still reported as one of the six temporal
  hits in Table 2 (Lin, log BF 1.35, q=0.008, declines after max) and named in
  the Results. ORIGINAL: `serine` (HILn_QI110) — linear time_from_max, richest of the 4
  (4 components vs 1). Uncontested, and it is one of only 6 temporal hits.
- [x] **M17. DONE — bilirubin is now Supplementary Figure S1.** `bilirubin` (HILp_QI19549) — the nonlinearity claim:
  `lin[hbi]` q=1.00 while `SE[hbi]` q=0.0092, i.e. a linear model misses it
  entirely. **This is the natural replacement for the lithocholate figure
  (M3)**, which made an SE claim the data no longer supports.
- [x] **M18. CHANGED — metronidazole is a TEXT result, not a figure.** The
  supplemental figure slot went to bilirubin (M17), which carries the
  nonlinearity claim with a positive log BF. Metronidazole's positive-control
  role is made explicitly in the Results instead (M21), and it appears as a
  bar in Figure 5 labelled (SE). This is a deliberate change from the original
  plan, not an oversight. ORIGINAL: `metronidazole` (HILp_QI2850) — supplemental positive control,
  SE:hbi log_bf=4.90 q=0.0092. Confounding by indication (antibiotic for
  active Crohn's), so it validates the method rather than the biology.
- [x] **M19. DONE — nervonic acid is the temporal showcase.** Chosen over
  serine because it carries a nonlinear (SE) temporal claim and because
  C24:1 SM is independently significant for HBI, so the figure illustrates a
  sphingolipid class the cross-sectional results support rather than a
  singleton. Its negative log BF (-1.07) is handled by the Methods paragraph
  on reading log BF against q, and stated in the Table 2 caption.
  ORIGINAL: **Consider a temporal showcase to replace oxalate (M2).** Serine
  (M16) already covers linear time_from_max. If a temporal *nonlinear* panel
  is wanted, the only candidates are nervonic acid (log_bf=-1.07) and
  NH4_C52:6 TAG (-2.80), both with negative log_bf — see M20.

## F. Presentation problems to decide on

- [x] **M20. DONE — dissolved by the M9 figure change, plus a Methods
  paragraph.** The figure no longer colours by a quantity that can contradict
  the significance claim, and the Methods explain the log BF / q distinction
  directly. ORIGINAL: **29 of 168 significant HBI components have NEGATIVE log_bf**
  (min -5.47, median +2.95). A component can clear BH while dropping it
  *improves* BIC, because significance is judged against the permutation null,
  not against zero. The heatmap colors by log_bf, so these render as evidence
  *against* an association they are reported as supporting. Options: report
  q and log_bf side by side; restrict figures to log_bf>0; or state the
  distinction explicitly. **Methodology is frozen — this is a reporting
  decision, not a criterion change.**
- [x] **M21. DONE.** Adrenate (C18n_QI43) is a known artifact. It reaches SE:hbi
  significance on a floor-variance (dead) component. It is separately
  significant via lin:hbi, so the metabolite-level count of 167 is unaffected,
  but **do not report "3 nonlinear HBI associations" — it is 2**
  (metronidazole, bilirubin). See TODO A1.
- [x] **M22. DONE for the figure** (bars at the floor print as `q <= 0.0003`,
  a bound). Still to check elsewhere in the text. ORIGINAL: **Ties at the resolution floor.** Many HBI hits share the minimum
  attainable q (q=0.0003 at B=100), so they cannot be ranked against one
  another. Report those q values as bounds, and do not describe any of them as
  "the most significant".
- [x] **M23. DONE — in the manuscript at two places.** Going B=60 -> B=100 lost 6
  hits and gained 4; the count is stable, the membership is not. Manuscript
  claims should concern the population of hits, not named borderline
  metabolites. (FINDINGS 27-28.)

  Recorded in `waveome_revision_tracker.md` under R1.M5/R2.5 (TODO C2, done).
  Concretely, the text may say "167 metabolites associate with HBI" but must
  not name a metabolite whose q sits within a few percent of 0.10 as though
  the cutoff were sharp. The chosen figures are safe: sorbitol log_bf 27.83,
  and neither nervonic acid nor bilirubin is near the boundary.

---

## Applied to `sn-article-revised.tex` (2026-09-22)

M12-M14 replaced the single "Significance criterion and reproducibility"
subsection with four: **Model specification**, **Significance criterion**
(six paragraphs: component evidence, null distribution, draw allocation,
p-values, multiplicity, calibration, and how to read log BF against q),
**Between-participant associations**, and **Reproducibility**.

The old rule is named and retired explicitly rather than quietly dropped --
"That rule is not a test: it thresholds a shrinkage estimate, carries no null
distribution, and controls no error rate" -- because R1.M5/R2.5 asked about
it directly and a reviewer will look for the acknowledgement.

The between-participant section reports a NULL result (0/564 for both
covariates) and draws the consequence: the cross-sectional associations are
within-participant associations, and the paper makes no claim that metabolite
levels separate participants by average disease activity. That is a real
narrowing of scope and should be read as such.

Every number verified against the data at edit time: 168 significant HBI
components, 29 of them with negative log BF, 2256 tests, B1=100, 41%/47%
between-participant variance for HBI / days-from-max.


M3 replaces Figure 6: `lithocholate_hbi.png` -> `sorbitol_hbi_conditional.png`
(`\label{fig:sorbitol_ids}`). The old caption asserted "a common squared
exponential HBI kernel where higher values of HBI are associated with lower
lithocholate" -- lithocholate's SE[hbi] is now q=1.00, and the one lithocholate
feature that is significant sits on lin[hbi] at log_bf -1.01, too weak to carry
the sentence. Sorbitol states the same *kind* of claim (individual offset plus a
common covariate kernel) on evidence that holds: log_bf 27.8, q=0.001, the
largest log Bayes factor of the 167 HBI hits, and its live components are
exactly cat[participant_id] 0.377 + lin[hbi] 0.107. Note the kernel is LINEAR,
so the caption no longer claims a nonlinear HBI effect.


M1, M2, M4, M5, M6 are in the working copy; `sn-article.tex` is untouched.
Compiles clean (`pdflatex`, exit 0, no errors).

**Figure added:** `figures/nervonic_acid_parts_output.{png,pdf}` replaces
`oxalate_parts_output.png` as Figure 7 (`\label{fig:nervonic}`).

**Relationship column re-derived**, not carried over: each entry is read off
the corrected additive decomposition over +/-60 days around the maximum. The
old descriptions came from the buggy decomposition (FINDINGS 29) and from
metabolites that are no longer significant.

**Citations now unused** after M1/M2 -- safe to leave, or prune from the .bib:
`kaczmarczyk_altered_2022`, `parada_venegas_short_2019`,
`xu_characterization_2022`, `bai_bile_2024`, `gkentzis_urolithiasis_2016`,
`jose_extraintestinal_2008`, `li_gut_2022`, `liu_microbial_2021`,
`siener_intestinal_2024`, `thomas_emerging_2022`.

**Flagged for the authors, not resolved:**
- Two of the six temporal hits (`NH4_C56:2 TAG`, `NH4_C52:6 TAG`) are
  annotated *redundant ion* in the source metabolomics table. The table and
  Discussion now say so, but whether they are independent findings is a
  chemistry call.
- The functional-class list is deliberately vague ("several classes...
  lipids (lysophospholipids and sphingomyelins prominent among them)")
  pending the curated re-derivation in M8. Naming a definitive set needs
  chemical annotation, not regex on metabolite names.

### M13 language softened (2026-09-23)

The between-participant null is power-limited, not strong: 49 participant-level
points, and the largest correlation seen (Spearman |rho| = 0.45, nominal
p = 0.002) does not survive correction across 564 metabolites. The text now
reads it as "an absence of detectable between-participant signal at this sample
size rather than as evidence that none exists", which is both softer and more
accurate than the original wording.

### M8 blocker found: the reported count is FEATURES, not compounds

| | features | distinct names | distinct HMDB ids |
|---|---|---|---|
| HBI hits | **167** | **154** | 152 |
| days-from-max hits | 6 | 6 | **4** |

13 metabolite names appear twice among the HBI hits (alanine, arginine,
proline, leucine, phenylalanine, C16:0 SM, C18:0 LPC, C18:1 LPC,
glycocholate, ... ) -- the same compound measured as separate features, in
most cases on different chromatography methods. Two of the six temporal hits
have no HMDB id at all ('redundant ion').

**RESOLVED (author decision): report both.** The Results now read "167
metabolite features --- corresponding to 154 distinct annotated compounds,
since some compounds are measured as more than one feature". The temporal
sentence says "six metabolite features ... resolve to four uniquely
identified compounds", and the Discussion caveat compares "six features
against 167, only four of them uniquely identified".

Annotation quality for the class re-derivation: 164 of 167 carry a real HMDB
id, but 39 of those are starred, i.e. representative ids standing for a class
of isomers rather than a specific compound.

### M21 resolved (2026-09-24): nonlinearity reframed and named

Author decision on all three points.

1. **Biology vs pharmacology separated.** The two nonlinear HBI associations
   are of different kinds and the Results now says so. Bilirubin is the
   biological result (lin q=1.00, SE q=0.009 -- a linear model misses it).
   Metronidazole is an antibiotic for active Crohn's, so its rise at high HBI
   reads as confounding by indication; it is reported as evidence the method
   recovers real structure it was not told to look for, since a prescribing
   threshold is exactly the abrupt non-monotone feature a linear term cannot
   represent. Not as a biomarker.
2. **Both named** in the Results, along with adrenate as the
   variance-collapsed third that is not counted.
3. **Bilirubin added as Supplementary Figure S1**
   (`figures/bilirubin_hbi_parts.png`). The supplement previously had no
   figures, so `\setcounter{figure}{0}` plus
   `\renewcommand{\thefigure}{S\arabic{figure}}` was added at the start of
   the Supplementary section -- otherwise it would have printed as "Figure 12"
   while the text called it "Supplementary Figure". Verified in the compiled
   PDF: the text reads "Supplementary Figure S1" and the caption "Fig. S1".

---

## Why Figure 5 changed from a heatmap to a bar chart (for the response letter)

The submitted Figure 5 was a kernel x metabolite heatmap of the top twenty
metabolites, coloured by log Bayes factor. It is replaced by a diverging bar
chart of effect sizes. Three independent reasons, any one of which would have
forced a change:

1. **The kernel axis no longer separates anything.** Under the permutation
   criterion 165 of 168 significant HBI components are linear and 3 are
   squared exponential, and exactly one metabolite (adrenate, the
   variance-collapsed artifact) is significant on both. One row of the
   heatmap was empty. In the submitted version, with 72 metabolites selected
   by a variance threshold, both kernels contributed and the axis was
   informative.

2. **No colour scale worked.** Colour had to encode something.
   - *log Bayes factor*, as submitted: 29 of the 168 significant components
     have a negative log BF, and the notebook drew them with `vmin=0`, so
     they were clamped to the bottom of the scale and rendered as though
     they carried no evidence at all. Removing the clamp is worse, not
     better: the figure would then colour significant findings as evidence
     against themselves.
   - *q-value*: no dynamic range. 36% of hits tie at q=0.0003 and there are
     35 distinct values in total across 0.0003-0.068. The figure would be a
     near-uniform block, and at the floor q is a bound rather than an
     estimate, so the colour would not mean what it appeared to mean.

3. **Effect size answers the question the figure is actually asked.** A
   reader looking at "metabolites associated with disease severity" wants to
   know which ones change and by how much. Significance now decides
   membership and effect size decides bar length, which separates the two
   questions instead of conflating them in one channel.

**Why log2 fold-change specifically.** The likelihood is negative binomial
with a log link and the kernel components are additive on that latent scale,
so a component's contribution is multiplicative on the response: the
fold-change is exactly the component's effect and does not depend on where
the other covariates are held. log2 rather than a raw ratio because it is
symmetric -- a doubling is +1 and a halving is -1 -- which a raw ratio is not
(2x against 0.5x), and an asymmetric quantity would distort a diverging
scale.

**Why the 5th-95th percentile range and not the full range.** The quantity is
the fitted contribution at the top of the range minus the contribution at the
bottom, in log2 units. Over the FULL observed range that is HBI 0 to 18 --
but only one observation sits at 18, against a median of 2. Full-range
fold-changes therefore extrapolate into territory a single data point
supports and inflate magnitudes about 2.5x (median |log2FC| 2.85 against
1.16). The ordering is essentially unaffected (Spearman 0.967, 19 of the top
20 in common), so restricting to HBI 0-7.1 costs nothing and stops the figure
overstating effects.

**Residual caveat, stated in the caption.** For a squared exponential
component an endpoint difference can understate an interior peak. Bars are
linear components unless marked (SE); metronidazole is the one SE bar in the
current selection.

### M23 in the manuscript (2026-09-25)

Two placements, because the fact and its consequence belong in different
sections.

**Methods, after Calibration** -- the technical statement: raising the budget
from 60 to 100 draws lost six features and gained four (~6% churn) while the
count barely moved (166 -> 165), and every feature lost had a live component
(variance 0.003-0.187), so these were marginal calls crossing the threshold
rather than artifacts being cleaned up. Concludes that the count is more
stable than boundary membership.

**Discussion limitations** -- the consequence for reading the results: report
a population of associated metabolites, not a definitive roster, and do not
single out individual borderline features.

One correction made during the edit. The limitations sentence first read "the
metabolites we highlight in figures were chosen well clear of that boundary",
which is false for Figure 5: the bar chart is selected by EFFECT SIZE among
all significant features and therefore reaches q=0.0684. It now names the
three individually-shown metabolites with their q-values (sorbitol 0.001,
nervonic acid 0.016, bilirubin 0.009) and says explicitly that Figure 5 spans
the full range of admitted evidence, which is why every bar carries its own q.

### M15-M19 swept (2026-09-25)

All four were written against the candidate set from before the figures were
chosen (proline / serine / bilirubin / metronidazole). The final set is
different, so these close as superseded rather than as work performed.

Verified consistent across manuscript and notebook:

| slot | figure | produced by |
|---|---|---|
| Fig. 5 | `ihmp_hbi_effect_sizes.png` | notebook cell 15 |
| Fig. 6 | `sorbitol_hbi_conditional.png` | cell 21, `CONDITIONAL` |
| Fig. 7 | `nervonic_acid_parts_output.png` | cell 21, `SHOWCASE` main |
| Fig. S1 | `bilirubin_hbi_parts.png` | cell 21, `SHOWCASE` supp |

Mentions in `sn-article-revised.tex`: proline 0, serine 3, metronidazole 2,
sorbitol 6, nervonic acid 6, bilirubin 5. Proline is correctly absent; serine
and metronidazole survive as text results rather than figures.
