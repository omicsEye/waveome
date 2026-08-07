"""Visual validation of the floor-anchor + non-floor-fold-spread null.

Two diagnostics per (kernel_type, covariate) stratum:

  Figure 1 -- fit check. Histogram of the NON-FLOOR components' log_bf (the
  population where the null-spread claim is actually testable; the near-floor
  components are a point mass at `loc` by construction and all receive
  p=0.5). Overlaid with the fitted Normal(loc, sigma_null) density, scaled to
  2*n_below: if the below-anchor values are the left half of a symmetric
  null, the implied total null count among non-floor components is twice the
  observed below count. Mass on the right in EXCESS of that curve is the
  real-signal component the test is meant to detect.

  Figure 2 -- normality check. The spread estimate uses only the below-anchor
  values folded around `loc`, corrected by 1/sqrt(1-2/pi) -- a correction
  valid only if those folded values are half-normal. QQ-plots them against
  half-normal(sigma_null) quantiles, with Shapiro-Wilk on the symmetrized
  (+/-) sample as a numeric companion.

Writes output/null_fit_check.png and output/null_qq_check.png.
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, shapiro

from waveome.model_search import _component_covariate_names
from waveome.utilities import VAR_CUTOFF_DEFAULT

INPUT_FP = (
    "output/fit_penalized_models_revision_full_scipy_ls_prior_no_prune_clamp_removed.pkl"
)

# Validated categorical pair (dataviz validate_palette.js, light mode: all
# checks pass). Gray is a neutral ink token for the observed data, not a
# third categorical identity.
C_NULL = "#2563eb"   # fitted null model
C_ANCHOR = "#d97706"  # floor-derived location
C_OBS = "#64748b"     # observed components


def build_df():
    with open(INPUT_FP, "rb") as f:
        gps = pickle.load(f)
    rows = []
    for name, model in gps.models.items():
        detail = getattr(model, "feature_importance_detail", None)
        if detail is None:
            continue
        kts, cns = _component_covariate_names(model.kernel_name, gps.feat_names)
        is_sum = model.kernel.name == "sum"
        terms = model.kernel.kernels if is_sum else [model.kernel]
        for idx, (kt, cn, comp) in enumerate(zip(kts, cns, detail[:-1])):
            lb = comp["log_bf"]
            if lb is None or not np.isfinite(lb):
                continue
            va = getattr(terms[idx], "variance", None)
            if va is None:
                continue
            rows.append({
                "stratum": f"{kt}:{cn}",
                "log_bf": lb,
                "near_floor": float(va.numpy()) < VAR_CUTOFF_DEFAULT,
            })
    return pd.DataFrame(rows)


def fit_stratum(g):
    """Return (loc, sigma_null, below, non_floor) or None if not estimable."""
    fl = g[g["near_floor"]]
    if len(fl) < 3:
        return None
    loc = fl["log_bf"].mean()
    non_floor = g[~g["near_floor"]]
    below = non_floor.loc[non_floor["log_bf"] < loc, "log_bf"].values
    if len(below) < 2:
        return None
    sigma = np.std(loc - below) / np.sqrt(1 - 2 / np.pi)
    return loc, sigma, below, non_floor


def main():
    df = build_df()
    strata = sorted(df["stratum"].unique())

    fits = {}
    for s in strata:
        f = fit_stratum(df[df["stratum"] == s])
        if f is not None:
            fits[s] = f
    keys = [s for s in strata if s in fits]

    # ---- Figure 1: does the fitted null explain the observed bulk? ----
    fig, axes = plt.subplots(4, 3, figsize=(13, 12))
    for ax, s in zip(axes.ravel(), keys):
        loc, sigma, below, non_floor = fits[s]
        g = df[df["stratum"] == s]
        n_floor = int(g["near_floor"].sum())
        vals = non_floor["log_bf"].values

        lo = min(vals.min(), loc - 4 * sigma)
        hi = np.percentile(vals, 99) if len(vals) > 20 else vals.max()
        hi = max(hi, loc + 4 * sigma)
        bins = np.linspace(lo, hi, 40)
        ax.hist(vals, bins=bins, color=C_OBS, alpha=0.55,
                label=f"observed non-floor (n={len(vals)})")

        binw = bins[1] - bins[0]
        xs = np.linspace(lo, hi, 400)
        implied_null_n = 2 * len(below)
        ax.plot(xs, implied_null_n * binw * norm.pdf(xs, loc, sigma),
                color=C_NULL, lw=2,
                label=f"fitted null N({loc:.2f}, {sigma:.2f}²)")
        ax.axvline(loc, color=C_ANCHOR, lw=2, ls="--",
                   label=f"floor anchor ({n_floor} at atom)")

        ax.set_title(f"{s}\nn_below={len(below)}", fontsize=9)
        ax.set_xlabel("log_bf", fontsize=8)
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(alpha=0.15, lw=0.5)
    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    fig.suptitle(
        "Fitted null vs observed non-floor log_bf (excess right of the curve = signal)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("output/null_fit_check.png", dpi=130)
    print("wrote output/null_fit_check.png")

    # ---- Figure 2: are the folded below-anchor values half-normal? ----
    fig2, axes2 = plt.subplots(4, 3, figsize=(13, 12))
    print(f"\n{'stratum':36s}{'n_below':>9s}{'sigma':>8s}{'shapiro_p':>11s}")
    for ax, s in zip(axes2.ravel(), keys):
        loc, sigma, below, _ = fits[s]
        folded = np.sort(loc - below)
        n = len(folded)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theo = sigma * norm.ppf(0.5 + 0.5 * probs)  # half-normal quantiles

        # Shapiro on the symmetrized sample (+/- folded): tests the parent
        # normal the half-normal correction assumes.
        sym = np.concatenate([folded, -folded])
        sp = shapiro(sym).pvalue if len(sym) >= 3 else np.nan
        print(f"{s:36s}{n:9d}{sigma:8.3f}{sp:11.4f}")

        ax.scatter(theo, folded, s=18, color=C_OBS, zorder=3,
                   label=f"folded below-anchor (n={n})")
        hi = max(theo.max(), folded.max()) * 1.05
        ax.plot([0, hi], [0, hi], color=C_NULL, lw=2, ls="-",
                label="half-normal reference")
        ax.set_title(f"{s}\nShapiro p={sp:.3f}", fontsize=9)
        ax.set_xlabel(f"theoretical half-normal (σ={sigma:.2f})", fontsize=8)
        ax.set_ylabel("observed |log_bf − loc|", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(alpha=0.15, lw=0.5)
    for ax in axes2.ravel()[len(keys):]:
        ax.set_visible(False)
    fig2.suptitle(
        "QQ: folded below-anchor values vs half-normal "
        "(the 1/√(1−2/π) correction assumes this line)", fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    fig2.savefig("output/null_qq_check.png", dpi=130)
    print("\nwrote output/null_qq_check.png")

    # ---- Figure 3: how much does the scale estimator matter? ----
    # Three candidate sigmas for the same (validated) location. Line style
    # carries identity as well as hue -- the green/amber pair sits in the
    # 6-8 protan dE band, legal only with secondary encoding.
    cands = [
        ("std (current)", C_NULL, "-",
         lambda f: np.std(f) / np.sqrt(1 - 2 / np.pi)),
        ("MAD", C_ANCHOR, "--",
         lambda f: np.median(f) / norm.ppf(0.75)),
        ("std, top-10% trimmed", "#059669", ":",
         lambda f: np.std(f[:-max(1, int(0.1 * len(f)))]) / np.sqrt(1 - 2 / np.pi)),
    ]
    fig3, axes3 = plt.subplots(4, 3, figsize=(13, 12))
    for ax, s in zip(axes3.ravel(), keys):
        loc, _, below, non_floor = fits[s]
        fold = np.sort(loc - below)
        vals = non_floor["log_bf"].values

        lo = min(vals.min(), loc - 4 * fold.std())
        hi = loc + max(3.0, 4 * fold.std())
        bins = np.linspace(lo, hi, 36)
        ax.hist(vals, bins=bins, color=C_OBS, alpha=0.55, label="observed non-floor")
        binw = bins[1] - bins[0]
        xs = np.linspace(lo, hi, 400)
        for label, colour, ls, fn in cands:
            sig = fn(fold)
            ax.plot(xs, 2 * len(below) * binw * norm.pdf(xs, loc, sig),
                    color=colour, lw=2, ls=ls, label=f"{label}: σ={sig:.2f}")
        ax.axvline(loc, color="#94a3b8", lw=1.2, ls="-", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("log_bf", fontsize=8)
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(alpha=0.15, lw=0.5)
    for ax in axes3.ravel()[len(keys):]:
        ax.set_visible(False)
    fig3.suptitle(
        "Scale-estimator sensitivity: same validated location, three σ estimates",
        fontsize=11)
    fig3.tight_layout(rect=[0, 0, 1, 0.97])
    fig3.savefig("output/null_sigma_sensitivity.png", dpi=130)
    print("wrote output/null_sigma_sensitivity.png")


if __name__ == "__main__":
    main()
