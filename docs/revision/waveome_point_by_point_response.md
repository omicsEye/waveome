# waveome — Point-by-point response (working draft)

Companion to `waveome_revision_tracker.md`. Keyed to each comment in the reviewer document, in order.

**Conventions:**
- **Response** = reviewer-facing reply voice. Stable (text-only) items are near-final; items whose punchline is a number/figure carry an explicit `[RESULT PENDING: …]` marker to fill once the code/analyses run.
- `[loc]` = manuscript location pointer (section/figure/line) to be filled against the revised manuscript.
- **Change** = internal code/paper action and status tag (`[locked]` / `[planned]` / `[pending]`). Strip Change + tags from the version submitted to reviewers.

> Cross-cutting: the new statistic + selection rule run in **both** the simulation and the iHMP analysis, so headline numbers (72 metabolites, top-20 heatmap, Table 1, Fig 4 / sens-spec) will be **re-run and may move** before the response is finalized — hence the pending markers.

---

## Editor

**E1 — Address all comments; concerns on NB use, reporting discrepancies, data splitting, harder settings; contextualize novelty; do not overstate.**
- **Response:** We thank the editor and reviewers for the constructive and detailed assessment. In this revision we expand the simulation study with harder regimes and additional baselines, replace our selection criterion with a calibrated false-discovery-controlled procedure, clarify and justify the negative-binomial likelihood, correct the reporting and checklist discrepancies, and reframe the biological findings as hypothesis-generating. Throughout, we have aimed to state the contribution and its limitations plainly rather than overclaim. The checklist and Editorial Requests Table are handled separately from this point-by-point, per the editor's instruction.
- **Change:** All actions below. **[planned]**

---

## Reviewer 1 — Major

**R1.M1 — Methodological novelty not sharply established.**
- **Response:** We thank the reviewer for pushing us to sharpen this, and we agree that the individual building blocks — additive and compositional kernel search, sparse and variational GPs, negative-binomial likelihoods, and shrinkage-based selection — are established in the literature we cite. Our distinct methodological contribution is the placement of a horseshoe prior directly on each kernel component's *variance* hyperparameter. Because the variance is the one parameter common to every kernel, this extends shrinkage-based selection to a *heterogeneous* kernel vocabulary (categorical, linear, Matérn, periodic, polynomial, and squared-exponential), whereas prior penalized-GP selection operates on the lengthscale/relevance side of distance-based kernels and is therefore confined to squared-exponential/ARD forms — for example, Yi et al. (2011) apply LASSO-type sparsity penalties (and variants such as SCAD and adaptive LASSO) to the per-covariate ARD relevance/inverse-lengthscale weights, leaving the kernel variance unpenalised, under an exact (non-scalable) Gaussian GP, and Vo & Pati (2016) place a horseshoe on the squared-exponential bandwidth (with an L1 penalty on additive-component weights) under MCMC. Our approach further operates within a scalable variational framework with inducing points and supports non-Gaussian likelihoods and longitudinal, repeated-measures structure. We are careful not to claim priority for shrinking a component's scale as such (Vo & Pati shrink a variance-like component weight via L1); the novelty is the horseshoe-on-variance formulation and the heterogeneous-kernel, scalable, non-Gaussian, longitudinal setting it enables. We state this explicitly in a contribution paragraph at the end of the Introduction [loc], separating the infrastructure we integrate from the method we introduce.
- **Change:** New contribution paragraph; prior-work delimitation grounded in **primary-source-verified** characterizations of Yi 2011 (penalties — LASSO/ridge/bridge/SCAD/adaptive-LASSO — on the per-covariate ARD relevance weights `wq`, i.e. inverse-lengthscale; variance `v₀` **unpenalised**; exact GP; Gaussian regression + classification; cross-sectional) and Vo & Pati 2016 (horseshoe on SE bandwidth κ + L1 on component weights φ, MCMC, Gaussian, cross-sectional). Pivot the claim on *variance is common to all kernels; lengthscale/bandwidth is not* → heterogeneous-kernel generalization. **[locked]**

**R1.M2 — Simulation favorable to the proposed family; add harder settings and stronger baselines.**
- **Response:** We agree that our original simulations were favorable to GP-based methods, and we have substantially expanded them with five regimes, each targeting a specific assumption: **misspecified kernels** (the true function lies outside our kernel vocabulary), **non-GP change-point dynamics** (abrupt, non-smooth shifts that smooth kernels cannot represent), **stronger nuisance and confounding structure** (irrelevant-but-correlated covariates that stress false-positive control), **zero-inflation** (excess zeros that stress the likelihood), and **higher-dimensional noise covariates** (selection specificity as the number of irrelevant inputs grows). We have added GPcounts (which shares our GPflow framework and a negative-binomial/ZINB likelihood) as a baseline, and attempted to include LonGP as the closest conceptual competitor. `[RESULT PENDING: relative performance across the new regimes; note on whether LonGP integration succeeded or justified omission.]`
- **Change:** Wire `effects.py` DGP into the single-output harness; add misspecified-kernel + change-point generators; integrate GPcounts; attempt LonGP; run under the finalized selection rule. **[planned]**

**R1.M3 — Random 80/20 split leaks subject structure; report subject-wise and forward-in-time splits.**
- **Response:** We agree that a random split is inappropriate for longitudinal data in which subject identity is modeled, and we now report subject-wise splits [loc]. We have also added forward-in-time splits as a robustness check. We would note, respectfully, that our evaluation is aimed at interpolation and at inference about the data-generating process — for example, the peri-flare dynamics that motivate the application — rather than at forecasting, so a forward-in-time split speaks to a somewhat different question; we nonetheless agree it is informative and include it. These splits are simulation robustness checks; the iHMP analysis is not split, for the reasons given under Major comment 4. `[RESULT PENDING: subject-wise and forward-in-time results.]`
- **Change:** Subject-wise CV via `make_folds(unit_col=…)`; forward-in-time split in the sim harness. **[planned]**

**R1.M4 — Real-data NB likelihood needs justification; compare alternatives.**
- **Response:** We thank the reviewer; this prompted us to describe our modeling choice far more carefully. We confirm that the iHMP analysis used the negative-binomial likelihood applied directly to the continuous metabolite intensities, and we now present it as such: a flexible, over-dispersed, zero-supporting *continuous working likelihood* (the negative-binomial form is evaluated on the continuum via the gamma function), chosen for its mean–variance behaviour — variance growing with the mean and a heavy right tail — and its handling of exact zeros, which the log-normal and gamma models cannot both provide. We make this explicit in the Methods, including that the count mass function does not integrate to one over the non-negative reals (so the score is an unnormalised log-density, valid for estimation and for difference-based comparisons under a fixed likelihood, but not for absolute comparison across likelihood families). We support the distributional choice with the **simulation study**, where held-out KL-divergence cleanly compares the negative binomial against log-normal and zero-inflated alternatives. On the iHMP data we report the **overlap of selected metabolites** across these likelihoods as a robustness check, rather than an absolute cross-likelihood fit ranking. `[RESULT PENDING: simulation fit comparison; iHMP selected-metabolite overlap.]`
- **Change:** Marginal-distribution figure; iHMP refits under log-normal + ZINB; report selected-metabolite overlap (not a BIC fit horse-race); reframe wording from "count data." **Manuscript text:** add a likelihood paragraph to **Methods** stating the working/unnormalised-likelihood choice, its mean–variance+zeros rationale, the un-normalisation limitation, and which uses are valid (differences yes, absolute cross-likelihood no), with a one-clause echo in the **Discussion** limitations; keep "unnormalised" **out of the abstract** (the R2.2 over-dispersed-continuous reframe suffices there). Prefer "working likelihood"; use "quasi-likelihood" only as a gloss (it has a stricter `V(μ)`/quasi-score meaning we don't fully invoke). Commit: **no NB-vs-log-normal absolute-fit claim anywhere** in text or response. *Caveat driving this: on continuous data the NB density is un-normalised (count PMF on the reals doesn't integrate to 1), so absolute log-lik/BIC isn't comparable across families — within-likelihood differences (ΔBIC, selection) and the simulation KL are unaffected.* **[locked / planned]**

**R1.M5 — Variable selection and significance criteria too ad hoc; log Bayes factor computation unclear.**
- **Response:** We agree the previous criteria were ad hoc, and we have replaced the variance threshold with a calibrated, false-discovery-controlled procedure. Each component's evidence is summarized by a refit ΔBIC statistic (log Bayes factor ≈ −½·ΔBIC); its significance is obtained by comparing that statistic to an empirical null constructed separately for each (kernel, covariate) pair — separately, because different kernel–covariate combinations have different null distributions (a flexible kernel fits noise more readily than a rigid one), so a single pooled null would be miscalibrated. In simulation, where ground truth is known, the null is built directly from known-null components. On the real iHMP data we considered subject-level permutation but found the combined cost of refitting under many permutations prohibitive (an estimated 530+ core-hours before reduced-model refits are even included, for a single covariate's null at a practical replicate count); we instead use a hardened empirical-Bayes (Efron two-groups) local-false-discovery-rate procedure, fit to each (kernel, covariate) stratum's own negative-evidence tail, with a location correction so the fitted null is centered where the statistic's own construction implies rather than at zero. This yields one-sided p-values to which we apply the Benjamini–Hochberg procedure at a target false-discovery rate. We additionally give a precise definition of the log Bayes factor in the Methods [loc], and we distinguish three reported quantities: evidence (ΔBIC), significance (FDR-controlled selection), and magnitude (deviance explained). `[RESULT PENDING: realized FDR/FWER vs nominal under simulation; sensitivity to the target FDR.]`
- **Change:** Refit reduced models in `calc_feature_importance_components`, matching the refit optimizer to whichever optimizer fit the full model; correct `calc_bic` to the real BIC formula (`k·log(n) − 2·loglik`); unify `var_cutoff` into a numerical pre-filter; implement the hardened empirical-Bayes fallback with a null-location correction and per-(kernel, covariate) stratification (`GPSearch.get_significance_table`, `calc_hardened_eb_qvalues`); new Methods subsection. Permutation (G1) assessed and not pursued — see `waveome_revision_tracker.md` item 5. **[locked / planned]**

**R1.M6 — Biological claims ahead of validation.**
- **Response:** We agree, and we have reframed the Crohn's disease findings as hypothesis-generating throughout the Discussion [loc]. We have softened the language around novel candidates and added an explicit limitation noting the absence of external-cohort validation and the potential for medication- and diet-related confounding of the observed associations.
- **Change:** Discussion edits; limitation paragraph. **[locked]**

**R1.M7 — Software/scalability reporting incomplete for a toolkit paper.**
- **Response:** We agree that the original supplement described the computing environment without reporting measured performance. We now provide two complementary pieces: scaling curves for runtime and peak memory as functions of sample size and the number of inducing points, and a real-world operating point on the iHMP analysis (total runtime, peak memory, approximate per-metabolite cost, and iterations to convergence). `[RESULT PENDING: the measured scaling curves and operating-point figures.]`
- **Change:** Instrument the simulation sweeps (scaling curves) and the iHMP re-run (operating point) with `psutil`; no standalone benchmark. **[planned]**

**R1.M8 — Reporting inconsistencies vs the ML checklist.**
- **Response:** We thank the reviewer for catching this. The checklist overstated our validation; we have corrected it to reflect that cross-validation is used only within penalization tuning and that no fully independent external dataset was analyzed.
- **Change:** Revise checklist entries. **[locked]**

---

## Reviewer 1 — Minor

**R1.m1 — NB parameterization ambiguous; latent draw can be negative.**
- **Response:** We have clarified the parameterization [loc]. The latent function is mapped through a logarithmic link, so the mean is m = exp(f); a negative latent value therefore yields a small positive mean rather than an invalid one, and the response follows a negative binomial with mean m and variance m + α·m². We have corrected the ambiguous NB(L, α) notation in the simulation setup accordingly.
- **Change:** Notation fix; one Methods sentence. **[locked]**

**R1.m2 — Clarify holdout KL-divergence computation per comparator.**
- **Response:** We have promoted this detail from the supplement to the main text [loc]. Each comparator is evaluated using its full predictive distribution — Gaussian-family methods using the residual standard deviation and negative-binomial methods using the fitted dispersion parameter.
- **Change:** Main-text sentences (text only). **[locked]**

**R1.m3 — Explain the "log Bayes factor" used in Figures 6 and 8.**
- **Response:** As described under Major comment 5, we now define this quantity precisely as −½·ΔBIC for the refit drop-one-component comparison, with the attendant caveats stated, and we have updated the figure and table labels [loc] and the Methods to match.
- **Change:** Methods formula; relabel figures/Table 1. **[locked]**

**R1.m4 — Figure readability (fonts, small panels).**
- **Response:** We have standardized font sizes and labels across the figures and enlarged the smaller panels. `[RESULT PENDING: revised figures.]`
- **Change:** Update plotting scripts/utilities. **[planned]**

**R1.m5 — Separate predictive utility from interpretability.**
- **Response:** We agree and have tempered these claims [loc], presenting the variable-selection decomposition as an aid to interpretation illustrated through examples, rather than as independently validated interpretability.
- **Change:** Wording edits. **[locked]**

**R1.m6 — Report implementation details (versions, tolerances, init, seeds, fallback rates).**
- **Response:** We have added a reproducibility paragraph and table [loc] giving package versions, the optimizer and its tolerances, the initialization scheme, the random seed, and the rate of fit fallbacks/failures. `[RESULT PENDING: the fallback-rate figures.]`
- **Change:** Versions from `pyproject.toml`; fallback counts from logs. **[planned]**

**R1.m7 — Clarify roles of covariates (targets vs nuisance vs repeated-measure).**
- **Response:** We have added a table [loc] distinguishing the role of each covariate in the iHMP model: HBI and days-from-maximum-HBI are scientific targets; participant identifier and study site capture repeated-measure and nuisance structure; and race, sex, age, and time-in-study are adjustments, with time-in-study included to absorb seasonal variation. We also note that days-from-max and time-in-study are correlated by construction, so each term's contribution is assessed conditionally on the others.
- **Change:** Covariate-role table; sentence on target/adjuster correlation. **[locked]**

---

## Reviewer 2 — Main

**R2.1 — Why Python rather than R?**
- **Response:** We adopted Python primarily to build directly on GPflow and TensorFlow Probability, which provide the specific machinery the method requires: automatic differentiation, scalable variational and inducing-point GP inference, and the ability to compose custom likelihoods (negative binomial, zero-inflated negative binomial) with custom hyperparameter priors (the horseshoe) inside a single autodiff framework. The surrounding pipeline uses the standard Python scientific stack (NumPy, pandas, SciPy, scikit-learn) and Ray for parallel model fitting. We have added a sentence noting this rationale [loc].
- **Change:** One sentence. **[locked]**

**R2.2 — Disconnect between count-based simulation and continuous metabolomics application.**
- **Response:** We appreciate this observation; the disconnect is one of framing rather than of method, and we resolve it as described under Major comment 4. The negative binomial is the same likelihood throughout, used as a flexible over-dispersed continuous model. We have revised the specific "count data" claims in the abstract (e.g., "longitudinal count data with over-dispersion") and introduction so the through-line is over-dispersed, often zero-inflated, frequently continuous omics measurements modeled with a flexible likelihood — of which the negative binomial is the instance we use — and we now demonstrate the application under log-normal and zero-inflated negative-binomial likelihoods as well.
- **Change:** Abstract/intro instance of the R1.M4 reframe — revise the "count data" sentences in the abstract and Introduction; substance and analyses as R1.M4. **[locked / planned]**

**R2.3 — Temporal alignment is muddy; Fig 8 third subpanel is a very complex function.**
- **Response:** This is a fair and important point. We retain the maximum-HBI anchoring as an event-aligned frame for peri-flare dynamics, and we address the interpretability concern directly by re-plotting the temporal component over a **tighter window centered on the anchor** (days from maximum HBI ≈ 0), where the peri-flare interpretation is meaningful, and by stating explicitly that the function is not intended to be read far from the anchor. `[RESULT PENDING: re-cropped figure with the narrowed window.]`
- **Change:** Re-plot the temporal component over a tighter window around days-from-max ≈ 0; add the interpretability limitation in text. *(Optional, to assess empirically during the re-run, not promised: constraining the time-in-study adjuster to a less flexible kernel if it carries only slow seasonal drift — a fit change, not a figure tweak.)* **[planned]**

**R2.4 — Out-of-sample prediction for new subjects; handling of the categorical ID when unseen.**
- **Response:** For a subject not present in the training data, the categorical (random-intercept-like) kernel has no covariance with the unseen identifier, so that component reverts to the population mean and predictions for new subjects rest on the shared, non-identity components of the model. We have stated this explicitly [loc] and added a short illustration.
- **Change:** Confirm `Categorical` behavior on unseen levels; note/example. **[locked]**

**R2.5 — The notion of "significance" needs closer examination; threshold, sensitivity, FWER/FDR.**
- **Response:** We agree that our earlier use of "significant" did not match its conventional statistical meaning, and we have revised it throughout. Selection is now governed by the calibrated FDR procedure described under Major comment 5, so "selected" denotes control of the false-discovery rate rather than a p < 0.05 claim. We report sensitivity to the target FDR and, under simulation where ground truth is known, the realized family-wise error and false-discovery rates against their nominal levels. `[RESULT PENDING: the error-rate results.]`
- **Change:** As R1.M5; global wording pass on "significant." **[locked / planned]**

---

## Reviewer 2 — Minor

**R2.m1 — Fig 4 needs context: number of candidate features and true signal sparsity.**
- **Response:** We have added, in Section 2.1 [loc], the number of candidate (kernel, covariate) components and the number that are truly non-null, to provide context for the selection results.
- **Change:** Sentence in Results 2.1. **[locked]**

**R2.m2 — A combined score (e.g., F1) would help show trade-offs.**
- **Response:** We have added an F1 score alongside the sensitivity and specificity reporting to summarize the selection trade-off in a single measure. `[RESULT PENDING: the F1 results.]`
- **Change:** Add F1 to the simulation evaluation. **[planned]**

**R2.m3 — Why is the search-based method so conservative?**
- **Response:** The conservatism follows from the selection rule: BIC penalizes model complexity, and the ΔBIC ≤ 6 tolerance favors parsimonious structures, so the search controls false positives at some cost to power — consistent with the high specificity and lower sensitivity we observe. We have added this explanation [loc].
- **Change:** One sentence. **[locked]**

**R2.m4 — How well does it handle correlated covariates?**
- **Response:** Correlation between covariates affects attribution, not overall fit. Because shared signal can be credited to either correlated component, their individual contributions are weakly identified: each one's marginal deviance-explained is deflated, and a genuinely associated covariate can test non-significant when its effect is redundant with a correlated partner (so "not selected" can mean "redundant," not "null"). Importantly, this costs power and attribution but not error control — Benjamini–Hochberg remains valid under the positive dependence correlated covariates induce. We characterize this with a dedicated simulation. `[RESULT PENDING: how often each of the correlated pair is selected, and how deviance-explained divides between them.]`
- **Change:** Add a correlated-covariate simulation condition. Report two things from it: (1) **selection** — which member of the correlated pair is selected, and the selection frequency across replicates (right one / unpredictable split / neither); (2) **deviance split** — how the marginal deviance-explained divides between the two correlated components (illustrating the deflation). Ties to R1.m7 (covariate roles). **[locked / planned]**

**R2.m5 — State #subjects and obs/subject; minimum time points to fit a GP.**
- **Response:** We now report the number of subjects and the median observations per subject (five) where the data are introduced. On how many time points are needed, the requirement differs by effect, and our additive model draws on data at different levels: shared (population-level) effects pool across all subjects and are estimable even when each subject is sparsely sampled, provided subjects jointly cover the covariate range; individual-specific temporal trajectories require several observations *within* a subject (one identifies only the subject's offset, and resolving a smooth trajectory needs more, with uncertainty shrinking as the per-subject count grows); and the between-subject variance component requires enough subjects rather than many points each. A GP always returns a prior-regularized posterior, so fitting is not gated by a hard threshold — what improves with data is the identifiability of the lengthscale and variance and the width of the posterior. Because our simulation varies both the number of subjects and the observations per subject, the dependence of recovery on each is characterized empirically.
- **Change:** Text edit at data introduction; disaggregate the "minimum data" point by effect level (shared vs individual-specific vs between-subject). **[locked]**

**R2.m6 — Explain how HBI is computed and its clinical significance.**
- **Response:** We have added a brief description of the Harvey–Bradshaw Index [loc] — a clinical index of Crohn's disease activity that combines general well-being, abdominal pain, the number of liquid stools, and abdominal complications — and of its role as our primary severity measure.
- **Change:** 1–2 sentences (Harvey 1980 cited). **[locked]**

**R2.m7 — Did you really use NB for metabolite intensities? Resolve p13 vs p17.**
- **Response:** Yes; as clarified under Major comment 4 and Reviewer 1's minor comment 1, the negative binomial was applied to the continuous intensities as an over-dispersed, zero-supporting continuous likelihood. We have corrected the contradictory "treated as continuous outcomes" wording and justified the choice.
- **Change:** As R1.M4. **[locked / planned]**

**R2.m8 — Cite original horseshoe and kernel-search references.**
- **Response:** We agree the method subsections should attribute their origins. Duvenaud et al. (2013) was already cited in the Introduction; we now also cite it within the Search-based Kernel Selection subsection, which previously carried no attribution. For the penalization method, we add the original horseshoe references (Carvalho, Polson & Scott, 2009, 2010), which were missing, to the Penalization subsection, together with the additive-GP work of Vo & Pati (2016, arXiv:1607.02670). We distinguish our contribution from the latter explicitly: Vo & Pati place a horseshoe on the squared-exponential *bandwidth* and an L1 penalty on additive-component weights, within an MCMC-based, Gaussian, cross-sectional model, whereas we place a horseshoe on each kernel's *variance* hyperparameter — extending selection to heterogeneous kernel types — within a scalable variational framework supporting non-Gaussian likelihoods and longitudinal data.
- **Change:** Add citations *within* the Search and Penalization subsections (not just the intro): Duvenaud 2013 in Search (already in `.bib`/intro); Carvalho–Polson–Scott 2009 & 2010 and Vo & Pati 2016 (arXiv:1607.02670) are **new** — add to `sn-bibliography.bib` and the Penalization subsection. Delta verified against the full text of arXiv:1607.02670 (horseshoe on SE bandwidth κ; L1 on component weights φ; MCMC; Gaussian; cross-sectional). (Citation status verified against `\cite` keys in `sn-article.tex`: Duvenaud present in intro only; CPS and 1607.02670 absent everywhere.) **[locked]**

**R2.m9 — Prior on global τ missing; describe the selection step.**
- **Response:** We have added the prior on the global shrinkage parameter — a half-Cauchy on τ — to the model specification [loc]. We have also clarified that, because the horseshoe is a continuous shrinkage prior and does not itself yield discrete selection, selection is performed by the calibrated FDR procedure described under Major comment 5 rather than by a raw variance threshold.
- **Change:** Methods edit; confirm/expose τ prior in `set_penalization_factor`. **[locked]**

**R2.m10 — Several GitHub examples are "In Progress."**
- **Response:** We have revised the repository so that example analyses are either completed or clearly labeled as illustrative, and removed unfinished items, so that the released toolkit reads as complete.
- **Change:** Tidy `README.md`; gate/finish notebooks. **[planned]**
