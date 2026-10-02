# Independent review of `sn-article-revised.tex` (2026-09-25)

Produced by an independent review agent with access to both `.tex` files,
`MANUSCRIPT_CHANGES.md`, and the three result CSVs. Two axes were requested:
(1) narrative coherence and scientific rigour, (2) style consistency with the
submitted `sn-article.tex`.

## Verification of the agent's key factual claims

Three load-bearing claims were re-checked directly before triage. **All three
hold.**

| claim | verified |
|---|---|
| "nine metabolites" survives in the Discussion | YES — line 227 |
| SCFAs were screened out at B0=10, never tested at B1=100 | YES — butyrate/propionate/valerate all `n_draws=10`, `null_sd` ~1e-7, `log_bf` at the exact parameter penalty. Caproate IS tested at 100, as **two** features (q=0.2714 and q=0.2964); the text quotes only one |
| 14 tests retain a 60-draw budget | YES — `n_draws` distribution {10: 1422, 60: 14, 100: 820}; **0** of the 14 are significant |

The agent's summary that "the numbers are, with two exceptions, correct" is
consistent with our own M7 sweep; the two exceptions it found (the caproate
second feature, and the nervonic acid lag) are both real and both ours.

---

## BLOCKER

**1. Results says six temporal metabolites; Discussion says nine.**
Discussion line 227: *"identifying nine metabolites with significant dynamic
changes"*, immediately followed by a list of six. Results says six, Table 2
has six rows, and the same paragraph later says "six features against 167".
`MANUSCRIPT_CHANGES.md` marks M6 DONE — it is not. **This is our error: the
Results sentence was fixed and the Discussion sentence was missed.**
Fix: "identifying six metabolite features — four uniquely identified
compounds — with significant dynamic changes", then re-grep every spelled-out
count in the file.

## MAJOR

**2. Simulation results are pre-correction while Methods says "replaced
throughout".** Figs. 3-4 were produced under the old BIC and under the
variance>1e-4 rule that Methods line 451 now calls "not a test". Either
re-run, or narrow the sentence to "replaced it for the iHMP analysis" and
state which rule defines a waveome hit in the benchmark. (This is M11.)

**3. The paper never states how a waveome feature counts as "selected" in the
simulation benchmark.** All six comparators have explicit rules in the
supplement; `tab:comparators` has no waveome row. Pre-existing, but acute now
that the paper's central claim is that thresholding a shrinkage estimate is
not inference.

**4. The SCFA null quotes q-values from components screened out at 10 draws.**
Butyrate, propionate and valerate/isovalerate never reached B1=100; q=1.00 is
the screening branch's assignment on a degenerate null. The Discussion builds
a biological argument on it, which contradicts the paper's own line 465,
"untested is not the same as tested and null". Also: caproate has two
features (q=0.27 and q=0.30); only one is quoted.
Suggested replacement: *"the SCFA components collapsed to the variance floor
and carried no fitted effect (butyrate, propionate, valerate/isovalerate);
the two caproate features were tested and non-significant (q = 0.27, 0.30)."*

**5. The between-participant null appears only in Methods.** The most
consequential sentence in the revision is invisible to anyone reading
Abstract -> Results -> Discussion, while the Abstract still promises
"candidates for cross-sectional disease severity" unqualified. Reported in
the open it reads as rigour; in Methods only, as burial.

**6. Calibration and the permutation framework are absent from Abstract,
Introduction and Results — significant UNDER-claiming.** "permutation" and
"Benjamini" appear nowhere before line 155 and never in the Abstract. The
"three primary advances" list omits the one contribution most likely to win
acceptance. The revision reads as damage control rather than an improved
paper.

**7. Three different counts circulate (167 features / 168 components / 165
stratum) without reconciliation.** All three are individually correct; 168 is
introduced in Methods without derivation. Fix the vocabulary once: *features*
(167) vs *components* (168), and say "the linear-kernel HBI stratum" where
165 is meant.

**8. The functional-class claim is orphaned.** The heatmap that carried it is
gone; no figure or table now lists the 167 features or supports "clustered",
a word implying an analysis not performed. Fix: "span" instead of "clustered",
plus a supplementary table of all 167 features. The table also defuses #7.

**9. Methods narrates the revision history.** The self-denunciation of the old
rule is response-letter prose in the archival record. State the method
positively; move the critique to the response letter.

**10. Table 2 states a temporal lag the analysis does not support, and
contradicts the figure caption.** Table: "Peaks ~45 d before max"; caption:
"rises into the window around peak severity and decays afterwards".
`MANUSCRIPT_CHANGES.md` M10 records our own side-verification: the peak's 95%
interval is -148 to +25 days, **"Do not state a lag."** We stated it anyway,
to two significant figures, on the only nonlinear temporal finding.

**11. Sorbitol is the flagship figure and is also named as a possible
confounder** in a retained Discussion sentence, without acknowledging the
tension. The same paragraph does this properly for metronidazole.

**12. New prose is in a visibly different voice.** Rhetorical antithesis,
colon-driven appositives, self-commentary on the exposition. Objective
markers: `---` appears 12 times in the revision and 0 in the original;
`\emph{}` 5 times versus 0.

## MINOR

13. British spellings introduced into an American paper: favours, centred
    (x3), labelled (x2), grey (x2) — with "grey" adjacent to retained "gray"
    describing the same visual element.
14. Captions carry Methods content at 2-3x the length of retained captions.
    The q-annotation convention and (SE) marking are earned; the percentile
    rationale and Spearman 0.97 belong in Methods.
15. Math-mode numerals ($564$, $154$, $41\%$) against the original's plain
    integers — both forms now appear in the same paper.
16. `q=1.00` / `q=0.27` set outside math mode, unlike every other q.
17. "credible intervals" (new) vs "confidence intervals" (retained) for the
    same GP posterior band. The new text is *more* correct; propagate rather
    than revert.
18. Straight apostrophes in new text vs curly in retained (Crohn's x5 vs
    Crohn's x6).
19. `\paragraph{}` introduced as a structural level the paper never otherwise
    uses (7 instances; 0 in the original). Confirm it compiles as intended
    under `sn-jnl.cls`.
20. Mixed word/numeral in one clause ("One hundred sixty-seven ... 165 ... and
    two"); inconsistent hyphenation of squared(-)exponential.
21. Table 2 caption says B=100; 14 of 2,256 tests retain a 60-draw budget from
    an earlier run. **None is significant**, so nothing substantive is wrong,
    but the same CSV is the reproducibility artifact.

## NITPICK

22. nominal p = 0.002 quoted where the data gives 0.00160.
23. Eleven now-uncited bibliography entries.
24. Inconsistent HMDB zero-padding; "redundant ion" occupies a column headed
    "HMDB ID".
25. Metronidazole appears as a bar in the effect-size figure, unremarked,
    after the Results disclaim it as a biomarker.

## WHAT WORKS WELL

- **Numerical integrity is high.** ~25 quantities checked against three CSVs
  and the raw metabolomics table; no substantive error. Rounding conventions
  consistent.
- **The Methods significance section is genuinely strong** — null
  construction, two-stage draw allocation with B1 derived from the BH
  requirement, pooled-quantile p-values, within-stratum correction with a
  stated reason for not pooling, floored p reported as a bound, calibration
  with bootstrap CIs.
- **The negative results are handled with unusual integrity** — the
  between-participant 0/564, the SCFA null, declining to count adrenate's
  floor-variance SE, flagging redundant adducts, stating the draw-budget
  churn. Finding 5 is a complaint about *where* they appear, not *that* they
  do.
- **The bilirubin example is the right choice** — null linear term,
  significant SE term, positive log BF.
- **Retiring the headline findings was done properly** — oxalate, SCFAs,
  4-methylcatechol and taurolithocholate removed from every site including
  mechanism clauses and citation clusters; betaine explicitly relocated.
- **Figure/text coupling is complete** — every new figure referenced, no
  dangling references, old figure labels fully gone, all four image files
  present as PNG and PDF.

## Agent's suggested triage order

1 first (one word, and the kind of error that makes a reviewer distrust every
other number). Then 10 and 4 — both are claims the data does not support, both
cheap. Then 5 and 6 together as one pass, which is "where this revision stops
reading as a retreat". 2 and 3 need the HPC decision; the honest interim is a
narrowed sentence, not silence.

---

## Triage log

### FIXED 2026-09-25 — findings 1, 10, 4

**1 (BLOCKER).** Discussion line 227 now reads "identifying six metabolite
features --- four uniquely identified compounds --- with significant dynamic
changes". Swept every spelled-out count in the file afterwards; the remaining
ones (One hundred sixty-seven, Six metabolite features, Two of the six, six
features against 167, four of them) are all correct.

**10 (MAJOR).** Table 2 cell changed from "Peaks $\sim$45 d before max" to
"Elevated around max, declining after". The Fig. 7 caption was changed to
match and now states explicitly that the fitted maximum falls before the
peak-severity point but that its location is poorly determined and no claim
about a lag is made. This honours the side-verification in M10 (95% interval
-148 to +25 days) that the earlier text ignored.

**4 (MAJOR).** The SCFA sentence no longer quotes q-values for untested
components. It now distinguishes the two cases: the HBI kernel components of
butyrate, propionate and valerate/isovalerate collapsed to the variance floor
and were never tested, while the two caproate features were tested and were
not significant (q = 0.27 and q = 0.30).

Verified before writing: all three SCFAs have HBI kernel variances at exactly
1.0e-10, the optimizer floor. Tightened the wording from "the ... components"
to "the HBI kernel components" because propionate and valerate retain other
live components (4 and 1 respectively) -- only butyrate's model collapsed
entirely.

Also corrected: caproate's second feature (q = 0.2964) is now reported
alongside the first (q = 0.2714); the earlier text quoted only one of two.

pdflatex exit 0, no undefined references.

### FIXED 2026-10-01 — findings 5a, 5b(partial), 8

Minimal truthful set applied. The test used was: does the paper currently
state something untrue or actively mislead a reader who does not reach the
Methods? Three things qualified.

**5a.** Results now carries the between-participant null in the reader's
path: "These associations are established within participants. A
complementary test for between-participant differences, in which each
participant contributes a single summary observation, identified no
metabolite for either covariate (Methods)."

**5b.** Abstract scope named. "novel candidates for cross-sectional and
temporal disease severity" -> "novel candidates for disease severity and its
temporal dynamics, assessed within participants". In epidemiology
"cross-sectional" implies a between-subject comparison, which is exactly what
the data do not support.

**8.** "These metabolites clustered into several functional classes" ->
"span". No clustering analysis was performed; "clustered" asserted one.

**Finding 21 dropped after checking.** All six Table 2 rows, and all 168
significant HBI components, used B=100. The caption describes its own table
accurately. The 14 tests at a 60-draw budget exist only in the CSV and none
is significant, so no claim in the paper rests on them.

**Deliberately NOT applied: credit/framing edits.** The Abstract calibration
clause, the Introduction fourth advance, and the Results calibration
paragraph (findings 6c/6d/6e) are improvements to how the revision lands, not
corrections. Omitting a strength is not untruthful. Left for an author pass.

---

## OPEN — BLOCKS SUBMISSION

**Finding 2: "We have replaced it throughout" (Methods, L451) is false today.**

The simulation benchmark still selects waveome features by the variance
> 1e-4 rule, in the same paragraph that calls that rule "not a test".

Author decision (2026-10-01): leave the sentence as written, because the
simulation will be re-run with the permutation method and the sentence
becomes true at that point. This is recorded rather than fixed.

**The risk this carries.** If the simulation re-run slips or is descoped, a
false statement ships, in the Methods section the reviewers specifically
asked about (R1.M5/R2.5). Two things must both happen before submission:

1. the simulation sweep is re-run under the corrected BIC and the permutation
   significance criterion (see M11 -- the search variant's selections WILL
   change; 28-34 BIC units of movement against a 6-unit retention tolerance);
2. Figures 3-4 are regenerated and this sentence re-verified.

If (1) does not happen, the sentence must be narrowed to "We have replaced it
for the iHMP analysis" and the simulation Methods must state which selection
rule defines a waveome hit in the benchmark (finding 3, also still open).

### FIXED 2026-10-01 — findings 12-16, 18, 20 (style pass)

Author instruction: no `\emph` and no em-dashes anywhere in the manuscript,
and the revision must read in the same voice as the submitted version.

| marker | before | after | original |
|---|---|---|---|
| `---` (em-dash) | 8 | **0** | 0 |
| `\emph{}` | 4 | **0** | 0 |
| straight apostrophes in words | present | **0** | 0 |
| favours / centred / labelled / grey | present | **0** | 0 |

The 13 em-dash and `\emph` sites were rewritten rather than stripped, using
the constructions the original actually uses: parentheses for appositives,
commas for short interpolations, and separate sentences where the clause was
carrying its own argument. Three sentences were split in the process (the
log BF / q explanation, the follow-up-studies caveat, and the
between-participant power statement), which also brings the sentence lengths
closer to the original's.

Mechanical fixes alongside: American spellings restored; `q=1.00` and
`q=0.27` set in math mode like every other q; bare integers unwrapped from
math mode (564, 154, 49, 168, 2,256, 41\%, ...) to match the original's
"564 labeled metabolites".

**Three unicode em-dashes at L129, L218 and L220 were left in place.** They
are the original's own usage (`sn-article.tex` contains the same three), so
removing them would move the revision further from the submitted style, not
closer. Flagged for an author call rather than changed.

**Remaining en-dashes are correct typography**, not em-dashes:
`Benjamini--Hochberg` (x4), `kernel--covariate`, and the ranges
`0.884--0.893` and `1--4`.

### STILL OPEN after the style pass

- **17.** "credible intervals" (new sorbitol caption) vs "confidence
  intervals" (retained GP captions) for the same posterior band. The new term
  is the correct one, so fixing this properly means editing retained captions.
  Author call.
- **19.** `\paragraph{}` x7 in the Methods; the original uses
  `\subsubsection*{}` at that depth and never `\paragraph`. Compiles fine,
  but it is a structural level the paper otherwise does not use.
- **7, 11.** The 167 / 168 / 165 vocabulary, and the sorbitol
  flagship-vs-confounder tension.
- **22-25.** Nitpicks: p = 0.002 vs 0.0016, eleven uncited bib entries,
  HMDB zero-padding, metronidazole unremarked in Figure 5.
