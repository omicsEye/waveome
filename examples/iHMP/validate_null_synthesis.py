"""Validate the floor-location + non-floor-fold-spread null-estimation
synthesis against fresh (clamp-branch-removed) no-prune data, per
(kernel_type, covariate) stratum.

For each stratum:
  - "near-floor" components: full-model (pre-drop) fitted variance for that
    term is < VAR_CUTOFF_DEFAULT. These cluster to an extremely tight,
    near-deterministic log_bf on genuine refit -- used here as a
    near-noise-free estimate of the null LOCATION.
  - "non-floor" components: escaped the coarse VAR_CUTOFF_DEFAULT prefilter,
    so individually enriched toward real signal -- but the slice of this
    population that falls BELOW the floor-derived location is still
    informative about the null's SPREAD (folded + corrected by
    1/sqrt(1-2/pi), matching calc_hardened_eb_qvalues' existing recipe).

Prints per-stratum: n_floor, floor location (mean/std), n_non_floor,
n used for fold-spread, sigma_null, and resulting significance count at
q<0.05 via calc_empirical_pvalue + BH.
"""
import pickle

import numpy as np

from waveome.model_search import _component_covariate_names
from waveome.utilities import VAR_CUTOFF_DEFAULT, calc_bh_qvalues, calc_empirical_pvalue

INPUT_FP = "output/fit_penalized_models_revision_full_scipy_ls_prior_no_prune_clamp_removed.pkl"


def main():
    with open(INPUT_FP, "rb") as f:
        gps = pickle.load(f)

    rows = []
    for name, model in gps.models.items():
        detail = getattr(model, "feature_importance_detail", None)
        if detail is None:
            continue
        kernel_types, cov_names = _component_covariate_names(
            model.kernel_name, gps.feat_names
        )
        is_sum = model.kernel.name == "sum"
        terms = model.kernel.kernels if is_sum else [model.kernel]
        for idx, (kernel_type, cov_name, comp) in enumerate(
            zip(kernel_types, cov_names, detail[:-1])
        ):
            log_bf = comp["log_bf"]
            if log_bf is None or not np.isfinite(log_bf):
                continue
            term = terms[idx]
            var_attr = getattr(term, "variance", None)
            if var_attr is None:
                continue
            full_var = float(var_attr.numpy())
            rows.append({
                "metabolite": name,
                "kernel_type": kernel_type,
                "covariate": cov_name,
                "stratum": f"{kernel_type}:{cov_name}",
                "log_bf": log_bf,
                "full_var": full_var,
                "near_floor": full_var < VAR_CUTOFF_DEFAULT,
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"Total components: {len(df)}")

    all_p = np.full(len(df), np.nan)
    df["p_value"] = np.nan

    for stratum, grp in df.groupby("stratum"):
        floor = grp[grp["near_floor"]]
        non_floor = grp[~grp["near_floor"]]
        n_floor = len(floor)
        n_non_floor = len(non_floor)

        if n_floor < 3:
            print(f"{stratum}: n_floor={n_floor} < 3, skipping (too few to anchor location)")
            continue

        loc = floor["log_bf"].mean()
        loc_std = floor["log_bf"].std()

        below = non_floor[non_floor["log_bf"] < loc]["log_bf"].values
        if len(below) < 2:
            print(f"{stratum}: n_floor={n_floor} loc={loc:.3f} (std={loc_std:.5f}) "
                  f"n_non_floor={n_non_floor}, but <2 below-location values -- skipping")
            continue

        folded = loc - below  # fold around loc, same sign convention as existing recipe
        sigma_folded = np.std(folded)
        sigma_null = sigma_folded / np.sqrt(1 - 2 / np.pi)

        # Build null pool: floor cluster (repeated as point mass) + symmetric
        # Gaussian(loc, sigma_null) draws, matching calc_empirical_pvalue's
        # empirical-pool convention.
        rng = np.random.default_rng(9102)
        null_pool = rng.normal(loc=loc, scale=sigma_null, size=20000)

        obs = grp["log_bf"].values
        pvals = calc_empirical_pvalue(obs, null_pool)
        all_p[grp.index] = pvals

        qvals = calc_bh_qvalues(pvals)
        n_sig = int(np.sum(qvals < 0.05))
        print(
            f"{stratum}: n={len(grp)} n_floor={n_floor} loc={loc:.3f} "
            f"(floor_std={loc_std:.5f}) n_non_floor={n_non_floor} "
            f"n_below={len(below)} sigma_null={sigma_null:.3f} "
            f"n_sig(q<0.05)={n_sig}"
        )

    df["p_value"] = all_p


if __name__ == "__main__":
    main()
