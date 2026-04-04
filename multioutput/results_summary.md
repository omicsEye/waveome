# iHMP Multioutput GP Analysis: Results Summary

## Latent Factor Structure

The MOGP model identified 10 latent factors, each capturing a distinct source of metabolomic variation in the iHMP IBD cohort. The largest factor (`Cat(participant_id)`) captured stable inter-individual differences, functioning as a metabolomic random effect. Site-specific (`Cat(site_name)`) and demographic factors (`Cat(race)`, `Cat(sex)`) were cleanly separated, with the sex factor showing strong face validity — anserine, a skeletal muscle dipeptide, was the most male-associated metabolite, consistent with established sex differences in muscle mass and carnitine metabolism.

Kernel labels and column indices for the 10 factors:
```
(0, 'Cat(participant_id)')
(1, 'Cat(site_name)')
(2, 'Cat(race)')
(3, 'Cat(sex)')
(4, 'Cat(general_wellbeing)')
(5, 'SE(hbi)')
(6, 'SE(study_days)')
(7, 'Mat52(time_from_max)')
(8, 'Per(study_days)')
(9, 'SE(age)')
```

---

## Disease Activity Findings

Two factors captured IBD-specific biology:

**SE(hbi)** captured a continuous dose-response to inflammation, with bile acids and tryptophan metabolites showing the strongest loadings — consistent with bile acid malabsorption and kynurenine pathway dysregulation in active Crohn's disease.

**Mat52(time_from_max)** captured transient peri-flare metabolic dynamics, dominated by a sharp coordinated decrease in triacylglycerols and phosphatidylcholines around the peak disease activity event. This lipid malabsorption signature — encompassing over 80 TAG and PC species — represents one of the most coherent and novel findings of the analysis, reflecting impaired intestinal fat absorption during active inflammation.

### Collinearity Caveat

These two factors were partially correlated in their loading vectors:
- Raw loading correlation: r = 0.361
- Absolute loading correlation: |r| = 0.509

This indicates ~25% shared variance, expected given that `time_from_max` is derived from HBI and the two variables are empirically correlated within the dataset. Importantly:

- `SE(hbi)` requires metabolites to track inflammation *continuously across all HBI values* (dose-response)
- `Mat52(time_from_max)` captures *transient changes relative to the peak event* (temporal dynamics)

These are mechanistically distinct patterns. The partial separation achieved by the model, with biologically coherent loadings on each factor, argues in favor of the method's ability to disentangle overlapping clinical measures. This should be noted as a limitation in the paper but does not undermine the core findings.

---

## Age and Temporal Trends

The `SE(age)` factor identified a lipid aging signature dominated by PC plasmalogens, DAGs, and medium-chain fatty acids. Plasmalogens are known to decline with systemic oxidative stress and aging, and DAG accumulation has been linked to altered lipid metabolism in older adults.

The `SE(study_days)` and `Per(study_days)` factors captured long-term secular trends and potential seasonal variation respectively. These are more difficult to interpret in the absence of dietary or medication metadata.

---

## Sex Factor Notes

The `Cat(sex)` factor was coherent but contained some unexpected directions worth flagging:
- **TMAO higher in females**: counterintuitive — literature generally shows TMAO higher in males due to red meat consumption and gut microbiome differences
- **EPA higher in males**: women typically have higher ALA→EPA conversion efficiency
- **Gabapentin and carboxyibuprofen**: likely reflect sex-differential drug use in the cohort rather than endogenous biology

---

## Methodological Note

A `LogNormal(1.0, 0.5)` prior on kernel length scales (median ≈ 2.7 in standardized input units, 90% CI ≈ [1.1, 6.6]) was necessary to prevent length scale collapse during optimization. Without this prior, continuous kernels — particularly `SE(age)` — drifted to very short length scales producing non-smooth, uninterpretable latent trajectories. The prior choice had a meaningful effect on the interpretability of the age and temporal factors and should be reported as part of the model specification.

---

## Pathway Enrichment Analysis (GSEA)

Preranked GSEA (gseapy, 1000 permutations) was run per factor using signed W loadings as the primary analysis and absolute |W| loadings as a sensitivity check. Pathway annotations were derived from the HMDB XML crosswalk (149 metabolites with confirmed KEGG pathway membership) supplemented by KEGG REST API for compounds lacking HMDB pathway annotations. Lipid class fallback metabolites were excluded from enrichment testing due to coarse pathway assignment. Significance threshold: FDR < 0.25 (standard GSEA convention).

Signed loadings detected 58 significant factor-pathway pairs vs. 18 for absolute loadings — directional information substantially improves power. Results below are from the signed analysis unless noted.

### SE(hbi) — Inflammatory Dose-Response
- **Glycerophospholipid metabolism positively enriched** (NES=2.00, FDR=0.0) — glycerophospholipids increase continuously with HBI, consistent with inflammatory membrane remodeling and phospholipase activation
- Bile acids (primary bile acid biosynthesis) trended positive but did not reach FDR<0.25 — likely underpowered given only 16 members in the annotated set; the raw loading patterns remain consistent with bile acid malabsorption
- Steroid biosynthesis borderline negative — cholesterol-related metabolites decrease with inflammation

### Mat52(time_from_max) — Peri-Flare Transient Dynamics
- **Steroid biosynthesis negatively enriched** (NES=-2.02, FDR=0.008) — cholesterol/steroid metabolites drop sharply around peak disease activity
- **Starch and sucrose metabolism negatively enriched** (NES=-2.00, FDR=0.008) — carbohydrate metabolites also decrease peri-flare, suggesting broad nutrient malabsorption
- Glycerophospholipids and glycerolipids positively enriched — lipid species accumulate around the flare event (consistent with impaired fat absorption and/or inflammatory lipid release)
- Primary bile acids borderline negative (FDR=0.215) — directional consistency with the HBI factor but weaker signal

### SE(age) — Lipid Aging Signature
The strongest multi-pathway signal of any factor:
- Glycerolipids and glycerophospholipids increase with age
- Multiple amino acid degradation pathways **negatively enriched**: BCAA degradation (NES=-1.83), glycine/serine/threonine (NES=-2.00), arginine/proline (NES=-2.02), nitrogen metabolism (NES=-1.92) — consistent with age-related decline in protein turnover and sarcopenia

### SE(study_days) — Secular Trend
- Glycerophospholipids decrease over study duration
- Amino acid and nucleotide metabolism (alanine/aspartate, lysine degradation, purine metabolism) increases — possibly reflecting treatment effects or dietary changes over the longitudinal study period

### Cat(participant_id) — Individual Differences
- Glycerolipids strongly enriched (NES=2.75) — the dominant source of inter-individual metabolomic variation is glycerolipid composition
- Bile acids, fatty acid degradation, and glycerophospholipids negatively enriched — individuals with lower glycerolipid loading tend to have higher bile acid and phospholipid profiles, reflecting a compositional trade-off

### Cat(sex)
- **Histidine metabolism** (NES=1.96, FDR=0.010) — strongest sex-specific pathway; histidine-derived metabolites (including 1-methylhistidine, anserine) are male-enriched
- Arachidonic acid and alpha-linolenic acid metabolism enriched in one sex — consistent with known sex differences in omega-3/6 fatty acid conversion efficiency

### Absolute vs. Signed Comparison
14 factor-pathway pairs were significant under both approaches. Signed-only findings (44 pairs) reflect pathways where members load coherently in one direction — these are the biologically interpretable findings. Absolute-only findings (4 pairs) reflect pathways with mixed-direction loading that is nevertheless strong — these warrant closer inspection.

---

## GSEA Heatmap

![MOGP Factor × Pathway Enrichment Heatmap](output/mogp_gsea_heatmap.png)

*Signed GSEA NES scores per factor-pathway pair. Only cells with FDR < 0.25 are shown (grey = not significant). Positive NES (red) = pathway members cluster at top of loading ranking; negative NES (blue) = pathway members cluster at bottom.*

---

## Tier 3: Unnamed Feature Projection

### Motivation

The iHMP metabolomics dataset contains 81,271 unnamed features identified only by m/z, retention time, and LC method — no chemical name, no HMDB ID, no pathway annotation. Conventional metabolomics pipelines discard these entirely. The MOGP approach offers a principled path to recover biological information from them.

### Method

After fitting the MOGP on 459 named metabolites, the latent factor structure (H matrix, shape n_samples × n_factors) was extracted by back-projecting the posterior mean Fmu through the fitted W matrix. Each unnamed feature was then projected onto this fixed latent space via OLS regression:

```
W_unnamed = (H'H)⁻¹ H' F_unnamed
```

where F_unnamed is the offset-centered log-count matrix for unnamed features (using per-feature log-median offsets matching the model's NB parameterization). Per-feature R² measures how well the fitted latent structure explains each unnamed feature's variation across samples.

### Validation

The projection was validated by applying the same procedure to named metabolites (treating them as unnamed) and comparing projected W to fitted W. Correlations with fitted W: r=0.83 (Cat(participant_id)), r=0.81 (SE(study_days)), r=0.76 (Per(study_days)), r=0.70 (SE(hbi)), r=0.62 (Mat52(time_from_max)). These represent the ceiling of the approach given the SVGP approximation — sufficient for ranking and prioritization but not for precise individual-feature inference.

### Coverage

Of 41,967 unnamed features passing quality filters (CV ≤ 0.3):
- R² ≥ 0.1: 32,201 (76.7%)
- R² ≥ 0.2: 15,024 (35.8%)
- R² ≥ 0.3: 5,237 (12.5%)

### Key Findings

Several unnamed features load as strongly onto SE(hbi) and Mat52(time_from_max) as the top named metabolites. Top candidates include:

- **C18n_QI13594** (m/z=751.59, RT=11.31 min, C18-neg): |W_hbi|=0.72, |W_tfm|=0.87, R²=0.57 — co-loads on both disease factors, m/z consistent with a long-chain glycerophospholipid or bile acid conjugate
- **HILp_QI22782** (m/z=643.34, RT=2.90 min, HILIC-pos): |W_hbi|=0.80, R²=0.40 — strongest SE(hbi) loading of any unnamed feature
- **HILn_QI11501** (m/z=435.31, RT=3.74 min, HILIC-neg): |W_hbi|=0.67, |W_tfm|=0.64, R²=0.56 — dual-factor candidate

These represent prioritized targets for MS/MS-based structural identification.

### Projection Visualization

![Unnamed Feature Projection](output/unnamed_feature_projection.png)

*Each point is an unnamed metabolic feature projected onto the SE(hbi) × Mat52(time_from_max) loading plane. Color = R² (variance explained by fitted latent factors). Red points = top 5% loading on both disease factors simultaneously. Black × markers = named metabolites for reference. Features in the upper-right and lower-left quadrants co-load with both disease-associated factors and are the highest-priority candidates for targeted identification.*

### Tentative Identifications (5 ppm, GI-relevant filter)

Of 40 top candidates queried against the local HMDB XML mass index, 12 returned matches within 5 ppm that are plausible in GI/stool metabolomics. Results are divided by biological interpretation:

**Decreasing with both SE(hbi) and Mat52(time_from_max) — extend the malabsorption/bile acid story:**
- **HILn_QI11501 → Varanic acid** (C26H44O5, 1.38 ppm) — bile acid intermediate; single match; directly extends the bile acid malabsorption narrative to an unnamed feature
- **C8p_QI6646 → N-Linoleoyl Isoleucine** (C24H43NO3, 0.18 ppm, high confidence) — microbially-derived N-acyl amino acid with endocannabinoid-like properties; novel connection between microbial lipid-amino acid conjugates and IBD activity
- **C18n_QI13365 → DG(18:3/24:1)** (C45H80O5, 0.43 ppm) — diacylglycerol, RT=15.75 min; extends named DAG/TAG malabsorption signature
- **C18n_QI13780 → LacCer(d18:1/12:0)** (C42H79NO13, 1.45 ppm) — lactosylceramide; sphingolipid catabolism in IBD; decreases with inflammation
- **C18n_QI13746 → TG(14:1/20:4/14:1)** (C51H86O6, 1.05 ppm) — TAG species consistent with lipid malabsorption signature
- **HILn_QI6175 → Tetracosapentaenoic acid (24:5n-6)** (C24H38O2, 1.10 ppm) — very long chain PUFA decreasing with inflammation

**Increasing with SE(hbi) — potential inflammatory response metabolites:**
- **C18n_QI6379 → hesperetin 3'-O-sulfate** (C16H14O9S, 0.58 ppm) — gut microbial flavonoid transformation product
- **HILn_QI1907 → D-Glucuronic acid 1-phosphate** (C6H11O10P, 0.70 ppm) — glucuronidation pathway intermediate
- **HILp_QI16121 → Vitamin E Nicotinate** (C35H53NO3, 2.09 ppm) — tentative; anti-inflammatory compound

**Flagged as uncertain — should note in paper:**
- HILp_QI22782 → Steviobioside (4.18 ppm): plant glycoside, likely a mass coincidence
- HILp_QI17886 → Apigenin 6-C-glucoside: dietary flavonoid, tentative
- C18n_QI5919 → Firibastat (0.39 ppm): ACE2 inhibitor drug; iHMP metadata does not include antihypertensive medication tracking; biologically implausible in an IBD stool study — likely an isobaric compound sharing the C8H20N2O6S4 formula

### Paper Framing

The key argument: rather than discarding 81K unnamed features, the MOGP latent space acts as a biological coordinate system. Features that co-vary with known disease-associated metabolites — even without chemical identification — are candidate biomarkers. The projection reduces 41,967 quality-filtered unnamed features to a focused list of ~50-100 high-priority candidates, making targeted MS/MS identification tractable. Of these, several receive tentative structural identification consistent with the named metabolite findings — including a bile acid intermediate (Varanic acid) and a microbially-derived N-acyl amino acid — supporting the biological coherence of the projection approach.

---

## Limitations and Future Work

### 1. No baseline comparison (priority for submission)
The novelty claim requires a head-to-head comparison against simpler approaches (PCA, NMF, individual metabolite-HBI correlations) on the iHMP dataset itself. The simulation benchmarks in `methods.py` address this in a controlled setting but not on real data. A reviewer will ask whether MOGP factor loadings are more interpretable or more pathway-enriched than PCA components. This comparison should be added before submission.

### Layer 2 Precision-Recall Validation

The transfer method was validated by applying it to labeled metabolites (treating them as if unlabeled) and comparing predicted pathway to known KEGG membership. A prediction is "correct" if the predicted pathway appears in the metabolite's true pathway list.

| Threshold | Precision | N retained |
|-----------|-----------|------------|
| 0.00 | 0.31 | 328 |
| 0.10 | 0.31 | 324 |
| 0.15 | 0.39 | 232 |
| 0.20 | 0.61 | 92 |
| 0.25 | 0.69 | 13 |

Recommended threshold: 0.20 (61% precision, 92 metabolites retained). Assignments above this threshold should be labeled as "probabilistic pathway hypotheses" rather than confirmed annotations.

![Layer 2 Validation](output/layer2_validation.png)

*Precision (blue) and number of retained metabolites (red) at different normalized confidence thresholds. Validated on 328 labeled metabolites with known KEGG pathway membership.*

---

### 2. Projection validation is internal consistency, not out-of-sample
The validation of W_proj vs W_fitted (r~0.7 for SE(hbi), r~0.62 for Mat52(time_from_max)) measures internal consistency — how well the OLS projection recovers the fitted loading structure. It is not a true out-of-sample test because H was derived from Fmu, which was jointly trained on all 459 metabolites including the ones used for validation. A LOO correction is not meaningful here since each metabolite contributes ~1/459th of the total signal, negligibly changing H. A genuine out-of-sample test would require a second cohort, a held-out time point, or a different metabolomics platform measuring the same samples. The validation should be labeled as "internal consistency" in the paper and interpreted accordingly: the projection reliably recovers the fitted loading structure but its generalization to truly unseen metabolites is unknown. This is an acknowledged limitation of the approach.

### 3. W loading stability (addressed in notebook)
Multi-seed stability analysis (3 seeds, full 50K iterations each, Hungarian alignment + sign correction) shows the following per-factor correlations with the main fit:

| Factor | mean |abs| corr |
|--------|-----------------|
| Cat(participant_id) | 0.979 |
| Cat(site_name) | 0.109 |
| Cat(race) | 0.808 |
| Cat(sex) | 0.809 |
| Cat(general_wellbeing) | 0.730 |
| SE(hbi) | **0.940** |
| SE(study_days) | 0.937 |
| Mat52(time_from_max) | **0.821** |
| Per(study_days) | 0.846 |
| SE(age) | 0.602 |

8/10 factors are stable (mean |r| ≥ 0.7). The two exceptions are Cat(site_name) (r=0.109 — weakly identified, consistent with site being a minor confounder) and SE(age) (r=0.602 — moderate, consistent with irregular trajectories noted elsewhere).

**The two disease factors are robustly identified**: SE(hbi) at r=0.940 and Mat52(time_from_max) at r=0.821, meaning the latent structure they capture is not an artifact of random initialization.

**Stable high-loading metabolites** (|mean W| > 0.3, CV < 0.3 across seeds):
- SE(hbi): metronidazole, quinine, gabapentin, C32:1 PC, C36:1 PC plasmalogen, 5alpha-cholestan-3beta-ol (6 metabolites)
- Mat52(time_from_max): C50:5 TAG, metronidazole, C9 carnitine, C38:4 PC plasmalogen, C32:1 PC (5 metabolites)

**Important confounder note**: metronidazole (antibiotic) and gabapentin (neuropathic pain drug) are among the most stable SE(hbi) loaders. This reflects medication-disease severity confounding — sicker patients are more likely to be on these drugs — rather than endogenous metabolite biology. These should be excluded or flagged separately in any clinical interpretation. This is a known limitation of cross-sectional metabolomics without medication adjustment.

The relatively small number of stable individual high-loaders (5-6 per factor) is consistent with the GSEA finding that factors capture diffuse pathway-level signals rather than tight individual metabolite clusters.

![W Loading Stability](output/w_stability_plot.png)

*Mean W loading vs. standard deviation across 3 random seeds for SE(hbi) (left) and Mat52(time_from_max) (right). Red points = stable high-loaders (|mean| > 0.3, std < 0.15). Orange dashed line = std=0.15 threshold.*

### 4. GSEA power limited by annotation coverage
Only 149/459 named metabolites had KEGG pathway annotations. Many pathways had fewer than 10 members, reducing power. Broader annotation sources — ClassyFire chemical taxonomy (covers all named metabolites by chemical structure) or LIPID MAPS — would increase coverage substantially and may reveal enrichments missed by the current analysis.

### 5. SVGP approximation gap
Factor reconstruction R² = 0.862 rather than 1.0, meaning ~14% of Fmu variance lies outside the W column space due to the sparse GP approximation. This provides a theoretical ceiling for projection accuracy (~r=0.87 maximum) and should be reported alongside the observed validation correlations (r~0.7) to contextualize how much headroom remains.

### 6. No sensitivity analysis on number of factors
The model uses 10 factors matching the number of kernel components. Robustness of the SE(hbi) and Mat52(time_from_max) findings to 8 or 12 factors has not been tested. If key findings only appear with exactly 10 factors, that would be a concern.

### 7. HBI and time_from_max collinearity needs formal treatment
The partial correlation (r=0.509 absolute) between SE(hbi) and Mat52(time_from_max) loadings is noted as a limitation, but a reviewer may want a formal model comparison (ELBO with 9 vs 10 factors, or a demonstration that both factors have distinct predictive value) rather than a qualitative argument.

---

## Open Questions

- **Periodic kernel**: Verify that the fitted period length scale for `Per(study_days)` is interpretable (months, not days). If short, it may be aliased noise rather than true seasonality.
- **HBI vs. general_wellbeing overlap**: Both measure disease burden — their respective factors may share variance. Worth checking W column correlation between factors 4 and 5.
- **Age factor non-monotonicity**: The `SE(age)` latent trajectory remains somewhat irregular even with the tightened prior, possibly reflecting genuine non-monotonic lipid-age relationships confounded by age at diagnosis in this IBD cohort.
