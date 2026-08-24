"""Two manuscript panels showing waveome's additive decomposition.

One metabolite per tested covariate, chosen by a single stated rule:
among metabolites significant for that covariate (q<=0.10), the one whose
model has the richest additive decomposition (most components explaining
>5% of deviance). A stated rule matters here because significance alone
does not identify a unique metabolite -- 72 of the 140 hbi hits share the
same minimum q, and all 3 time_from_max hits tie at 0.0101, so ranking by
q would be an arbitrary pick among ties.

  hbi           -> HILp_QI578  (6 components, lin:hbi DE=0.263)
  time_from_max -> HILn_QI110  (4 components, lin:time_from_max DE=0.274)

Both selected components are LINEAR. No metabolite significant for either
covariate has any nonlinear (SE) structure in that same covariate -- every
SE:hbi and SE:time_from_max component among the 143 significant models sits
at log_bf=-4.8, DE=0.000. Nonlinearity is not demonstrable on these data
and is not claimed here.

Titles are read from the permutation table rather than hard-coded, so they
follow a re-run instead of drifting.
"""
import pickle
import re

import matplotlib.pyplot as plt
import pandas as pd

PKL = "output/fit_penalized_models_revision_full_scipy_ls_prior_no_prune_clamp_removed.pkl"
PERM = "output/ihmp_permutation_significance.csv"
TFM = "output/ihmp_permutation_tfm_b120.csv"
PANELS = [("HILp_QI578", "hbi"), ("HILn_QI110", "time_from_max")]

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})


def main():
    perm = pd.concat(
        [pd.read_csv(PERM).query("covariate != 'time_from_max'"), pd.read_csv(TFM)],
        ignore_index=True)
    mbx = pd.read_csv("data/iHMP_labeled_metabolomics.csv", low_memory=False)
    names = mbx.set_index("Compound")["Metabolite"].to_dict()
    with open(PKL, "rb") as f:
        gps = pickle.load(f)

    captions = []
    for compound, cov in PANELS:
        r = perm[(perm.metabolite == compound) & (perm.covariate == cov)
                 & (perm.kernel_type == "lin")].iloc[0]
        # 72 hbi hits share the floor q, so report it as a bound, not a point
        floor = perm[perm.stratum == r.stratum].q_value.min()
        qtxt = (f"q<={floor:.4f}" if r.q_value <= floor else f"q={r.q_value:.4f}")
        label = names.get(compound, compound)

        fig, axes = gps.plot_parts(
            out_label=compound,
            x_axis_label="study_days",
            figsize=(15, 12),
            num_cols_in_fig=4,
            reverse_transform_axes=True,
            residual_dict={"resid_type": "pearson", "residuals_on_y_axis": False},
            prune_before_plot=True,
        )
        # Annotate each component panel with its q-value. Without this the
        # most eye-catching panel in a figure can be a REJECTED component
        # (proline's SE[time_from_max] peak, q=1.00) with nothing saying so.
        lut = {(r_.kernel_type, r_.covariate): r_.q_value
               for r_ in perm.itertuples()}
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
            f"{label} ({compound}): linear {cov} association "
            f"(log_bf={r.log_bf:.1f}, {qtxt}, null SD={r.null_sd:.2f}, "
            f"B={int(r.n_draws)})",
            fontsize=9, y=1.02)
        plt.tight_layout()
        stem = f"output/showcase_{cov}_{compound}"
        for ext in ("png", "pdf"):
            fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {stem}.png / .pdf   log_bf={r.log_bf:.1f} {qtxt}")
        captions.append(
            f"{label} ({compound}) -- {cov}: linear component log_bf="
            f"{r.log_bf:.1f}, {qtxt} (within-subject permutation, B="
            f"{int(r.n_draws)}, null SD={r.null_sd:.2f}).")

    with open("output/showcase_panels_caption.txt", "w") as f:
        f.write(
            "Additive decomposition of two metabolite models.\n\n"
            "Each panel shows the fitted additive kernel components of a "
            "single metabolite model; the component for the covariate under "
            "test is the one named in the title. Metabolites were selected "
            "by a single rule: among those significant for the covariate "
            "(q<=0.10, within-subject permutation), the one whose model has "
            "the most components explaining >5% of deviance. Selection by "
            "q-value alone would not identify a unique metabolite, since 72 "
            "of the 140 hbi hits share the minimum attainable q and all "
            "three time_from_max hits tie at 0.0101.\n\n"
            + "\n".join(captions) +
            "\n\nBoth selected components are linear. Components without a "
            "q-value were not permutation-tested -- only hbi and "
            "time_from_max were -- and untested is not the same as tested "
            "and null.\n")
    print("wrote output/showcase_panels_caption.txt")


if __name__ == "__main__":
    main()
