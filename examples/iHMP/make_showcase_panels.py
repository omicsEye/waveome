"""Two manuscript panels showing waveome's additive decomposition.

One metabolite per tested covariate, chosen by a single stated rule:
among metabolites significant for that covariate (q<=0.10), the one whose
model has the richest additive decomposition (most components explaining
>5% of deviance). A stated rule matters here because significance alone
does not identify a unique metabolite: many hits share the same minimum q
(all time_from_max hits sit at the pooled-null resolution floor), so ranking
by q would be an arbitrary pick among ties.

  hbi           -> HILp_QI578  (6 components, uniquely richest of 166 hits)
  time_from_max -> HILn_QI110  (4 components, vs 1 for the runners-up of 5)

Both selected components are LINEAR, which is a property of these two
metabolites rather than of the data as a whole. Under the corrected
ELBO-based BIC there ARE significant nonlinear components elsewhere -- 4
SE:hbi and 2 SE:time_from_max, five of the six on live components with
fitted lengthscales 0.76-1.98 (FINDINGS 26). The earlier claim that
nonlinearity was not demonstrable here belonged to the superseded statistic.

Titles are read from the permutation table rather than hard-coded, so they
follow a re-run instead of drifting.
"""
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PKL = "output/fit_penalized_models_revision_full_scipy_ls_prior_no_prune_clamp_removed.pkl"
PERM = "output/ihmp_permutation_significance.csv"
TFM = "output/ihmp_permutation_tfm_b120.csv"
# (compound, covariate, n_cols). Three columns at a 7.2in width gives 2.4in
# panels, which is the narrowest that still fits a title like
# "squared_exponential[time_from_max]" and the categorical legends without
# them spilling outside their axes. Four columns (1.8in) overruns both.
# (compound, covariate, kernel_type, n_cols). kernel_type names which
# component the figure is built around -- the first two showcase LINEAR
# associations, the third a NONLINEAR one, which only became reportable
# under the corrected BIC (FINDINGS 26).
PANELS = [("HILp_QI578", "hbi", "lin", 3),
          ("HILn_QI110", "time_from_max", "lin", 3),
          ("HILp_QI2850", "hbi", "squared_exponential", 3),
          ("HILp_QI19549", "hbi", "squared_exponential", 3)]

# Sized for print, not for the screen. Figures are saved at exactly the
# width they will occupy in the journal, so the publisher never rescales
# them and the point sizes below are the point sizes that print. The
# previous 15x12in figure had to shrink 2.07x to reach a double-column
# width, which drove 9pt text to 4.3pt -- under every journal's floor.
FIG_WIDTH_IN = 7.2       # double-column (183mm); use 3.5 for single column
PANEL_ASPECT = 0.85      # plot-area height / panel width
TITLE_OVERHEAD_IN = 0.60  # 3-line panel title + x-label, per row
SUPTITLE_IN = 0.35
DPI = 600                # for the raster copy; the PDF is vector

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "savefig.bbox": "tight", "pdf.fonttype": 42,  # embed as TrueType, editable
})


def main():
    # The B=120 time_from_max top-up is an OPTIONAL second pass, not part of
    # the main run. Fall back to the base results when it is absent, and say
    # which was used -- previously its absence crashed the script, which made
    # regenerating figures between the main run and the top-up impossible.
    base = pd.read_csv(PERM)
    if os.path.exists(TFM):
        perm = pd.concat([base.query("covariate != 'time_from_max'"),
                          pd.read_csv(TFM)], ignore_index=True)
        print(f"time_from_max: using the B=120 top-up ({TFM})")
    else:
        perm = base
        b = int(base.query("covariate == 'time_from_max'").n_draws.max())
        print(f"time_from_max: top-up absent, using base results at B={b} "
              "-- q-values for that covariate are PROVISIONAL")
    mbx = pd.read_csv("data/iHMP_labeled_metabolomics.csv", low_memory=False)
    names = mbx.set_index("Compound")["Metabolite"].to_dict()
    with open(PKL, "rb") as f:
        gps = pickle.load(f)

    captions = []
    for compound, cov, ktype, ncols in PANELS:
        r = perm[(perm.metabolite == compound) & (perm.covariate == cov)
                 & (perm.kernel_type == ktype)].iloc[0]
        # 72 hbi hits share the floor q, so report it as a bound, not a point
        floor = perm[perm.stratum == r.stratum].q_value.min()
        qtxt = (f"q<={floor:.4f}" if r.q_value <= floor else f"q={r.q_value:.4f}")
        label = names.get(compound, compound)

        # plot_parts decides how many components survive pruning, so the row
        # count (and thus the height) is only knowable after the call.
        # Building at the final print width means no rescaling later.
        panel_w = FIG_WIDTH_IN / ncols
        fig, axes = gps.plot_parts(
            out_label=compound,
            x_axis_label="study_days",
            figsize=(FIG_WIDTH_IN, panel_w * PANEL_ASPECT * 3),
            num_cols_in_fig=ncols,
            reverse_transform_axes=True,
            residual_dict={"resid_type": "pearson", "residuals_on_y_axis": False},
            prune_before_plot=True,
        )
        # Drop slots plot_parts left empty; an unused axes still renders as a
        # blank framed box, which reads as a missing panel rather than as
        # spare grid.
        used = 0
        for ax in list(axes.flatten()):
            if ax.has_data() or ax.get_title():
                used += 1
            elif ax in fig.axes:      # plot_parts may already have dropped it
                fig.delaxes(ax)
        # Each row needs its plot area PLUS the three-line panel title and the
        # x-label beneath it; sizing on the plot area alone squashed the
        # panels flat and let the categorical legends overflow their axes.
        rows = int(np.ceil(used / ncols))
        row_h = panel_w * PANEL_ASPECT + TITLE_OVERHEAD_IN
        fig.set_size_inches(FIG_WIDTH_IN, rows * row_h + SUPTITLE_IN)
        # Annotate each component panel with its q-value. Without this the
        # most eye-catching panel in a figure can be a REJECTED component
        # (proline's SE[time_from_max] peak, q=1.00) with nothing saying so.
        # Filter to THIS metabolite first. Keying the lookup on
        # (kernel_type, covariate) alone let all 564 metabolites overwrite
        # each other, so every panel was annotated with whichever metabolite
        # happened to be last in the table -- which tagged serine's null
        # lin[hbi] as significant and its significant lin[time_from_max] as
        # null.
        pm = perm[perm.metabolite == compound]
        assert len(pm), f"no permutation rows for {compound}"
        lut = {(r_.kernel_type, r_.covariate): r_.q_value
               for r_ in pm.itertuples()}
        for ax in axes.flatten():
            m = re.match(r"^(\w+)\[(\w+)\]", ax.get_title())
            if not m:
                continue
            q = lut.get((m.group(1), m.group(2)))
            ax.set_title(
                ax.get_title() + ("\nnot permutation-tested" if q is None
                                  else f"\nq={q:.3g}"
                                  + ("  SIGNIFICANT" if q <= 0.10 else "")),
                fontsize=8)

        fig.suptitle(
            f"{label} ({compound}): "
            f"{'nonlinear' if ktype.startswith('squared') else 'linear'} "
            f"{cov} association "
            f"(log_bf={r.log_bf:.1f}, {qtxt}, null SD={r.null_sd:.2f}, "
            f"B={int(r.n_draws)})",
            fontsize=9, y=1.02)
        plt.tight_layout()
        stem = (f"output/showcase_{cov}_{compound}"
                + ("_SE" if ktype.startswith("squared") else ""))
        for ext in ("png", "pdf"):
            fig.savefig(f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {stem}.png / .pdf   log_bf={r.log_bf:.1f} {qtxt}")
        captions.append(
            f"{label} ({compound}) -- {cov}: "
            f"{'nonlinear (SE)' if ktype.startswith('squared') else 'linear'} "
            f"component log_bf="
            f"{r.log_bf:.1f}, {qtxt} (within-subject permutation, B="
            f"{int(r.n_draws)}, null SD={r.null_sd:.2f}).")

    with open("output/showcase_panels_caption.txt", "w") as f:
        f.write(
            f"Additive decomposition of {len(captions)} metabolite "
            f"model{'s' if len(captions) != 1 else ''}.\n\n"
            "Each panel shows the fitted additive kernel components of a "
            "single metabolite model; the component for the covariate under "
            "test is the one named in the title. Metabolites were selected "
            "by a single rule: among those significant for the covariate "
            "(q<=0.10, within-subject permutation), the one whose model has "
            "the most components explaining >5% of deviance. Selection by "
            "q-value alone would not identify a unique metabolite, since "
            "many hits share the minimum attainable q -- every time_from_max "
            "hit sits at the pooled-null resolution floor.\n\n"
            + "\n".join(captions) +
            "\nTwo percentages appear in each figure and they use different "
            "denominators. A component's DE is its share of what the MODEL "
            "explains (its gain over the null model), so components can sum "
            "past 100% -- each is measured by dropping that component and "
            "refitting, and non-orthogonal components re-absorb one "
            "another's shared credit. The residual panel instead reports the "
            "share of TOTAL deviance, and states the model's overall "
            "explanatory power so the component DEs can be placed on that "
            "scale: bilirubin's SE[hbi] at 56.8% of an 11.7% explained "
            "portion is roughly 6.6% of total deviance, not 56.8% of the "
            "metabolite.\n"
            "\nThe two linear showcases select linear components, which is a property "
            "of these two metabolites and not of the cohort: significant "
            "nonlinear components exist elsewhere (4 SE:hbi, 2 "
            "SE:time_from_max). Components shown without a q-value were not "
            "permutation-tested -- only hbi and time_from_max were -- and "
            "untested is not the same as tested and null.\n")
    print("wrote output/showcase_panels_caption.txt")


if __name__ == "__main__":
    main()
