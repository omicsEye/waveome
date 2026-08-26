"""Why do 93% of squared-exponential components land on log_bf = -4.8?

Survey finding (docs/revision/FINDINGS.md): 2,102 of 2,256 SE components
carry an identical log_bf of -4.8, including 243 of the 301 that explain
>30% of their model's gain over null. Deviance explained and log_bf are
uncorrelated among live components (spearman +0.03). Because SE:hbi and
SE:time_from_max are two of the four tested strata, "no significant
nonlinearity in iHMP" is currently indistinguishable from "log_bf cannot
measure nonlinearity here".

This separates three candidate mechanisms for each pinned component:

  COLLAPSED  the fitted SE variance sits at VARIANCE_FLOOR, so the
             component is numerically dead and its DE is spurious
  ABSORBED   its fitted contribution is reproducible from the other
             components (high R^2), so dropping it costs nothing because
             the rest supply the same shape
  ELBO-BLIND the component is live and NOT reproducible, the posterior
             predictive worsens measurably when it is dropped, yet the
             ELBO -- which BIC and therefore log_bf are computed from --
             does not move

Three groups are measured so the pinned cases have something to be
compared against: pinned with real DE (the puzzle), pinned with ~zero DE
(expected to be genuinely empty), and unpinned (the working case).

Run:  python diagnose_se_collapse.py [n_per_group]
"""
import pickle
import sys
import time

import gpflow
import numpy as np
import pandas as pd

from waveome.model_search import _component_covariate_names
from waveome.predictions import individual_kernel_predictions
from waveome.utilities import (
    VARIANCE_FLOOR,
    calc_deviance_explained,
    convert_data_to_tensors,
)

PKL = "output/fit_penalized_models_revision_full_scipy_ls_prior_no_prune_clamp_removed.pkl"
TABLE = "output/all_component_results_with_significance.csv"
OUT = "output/se_collapse_diagnosis.csv"
MASS_POINT = -4.8
SEED = 9102


def main(n_per_group=25):
    d = pd.read_csv(TABLE)
    se = d[d.kernel_type == "squared_exponential"].copy()
    se["pinned"] = se.log_bf.round(1) == MASS_POINT
    rng = np.random.default_rng(SEED)

    groups = {
        "pinned_high_DE": se[se.pinned & (se.deviance_explained > 0.10)],
        "pinned_zero_DE": se[se.pinned & (se.deviance_explained <= 0.001)],
        "unpinned": se[~se.pinned],
    }
    picks = []
    for label, g in groups.items():
        take = g.iloc[rng.choice(len(g), min(n_per_group, len(g)), replace=False)]
        for r in take.itertuples():
            picks.append((label, r.metabolite, r.covariate,
                          r.deviance_explained, r.log_bf))
        print(f"{label:16s} pool={len(g):5d} sampled={min(n_per_group, len(g))}")

    gps = pickle.load(open(PKL, "rb"))
    X = gps.X.to_numpy()
    n = X.shape[0]
    rows = []
    t0 = time.time()
    for i, (label, met, cov, de, log_bf) in enumerate(picks, 1):
        m = gps.models[met]
        kt, cn = _component_covariate_names(m.kernel_name, gps.feat_names)
        idx = [j for j, kc in enumerate(zip(kt, cn))
               if kc == ("squared_exponential", cov)]
        if not idx:
            continue
        j = idx[0]
        y = gps.Y[met].to_numpy().reshape(-1, 1)
        data = convert_data_to_tensors(X, y)

        kern = m.kernel.kernels[j]
        var = float(np.array(kern.variance))
        ls = float(np.array(kern.lengthscales).ravel()[0])

        # component contributions at the observed points
        C = np.column_stack([
            np.asarray(individual_kernel_predictions(
                model=m, kernel_idx=q, data=data, X=X)[0]).ravel()
            for q in range(len(kt))])
        tgt = C[:, j]
        others = [q for q in range(len(kt)) if q != j and C[:, q].std() > 1e-8]
        if others and tgt.std() > 1e-12:
            A = np.column_stack([np.ones(n)] + [C[:, q] for q in others])
            beta, *_ = np.linalg.lstsq(A, tgt, rcond=None)
            r2 = float(1 - ((tgt - A @ beta) ** 2).sum()
                       / ((tgt - tgt.mean()) ** 2).sum())
        else:
            r2 = np.nan

        def elbo(mod):
            return float(mod.maximum_log_likelihood_objective(data))

        def pred_ll(mod):
            mu, v = mod.predict_y(data[0])
            nl, ml, sl = calc_deviance_explained(
                model=mod, data=data, model_mu=mu, model_var=v,
                return_deviance_explained=False, aggregate=False,
                return_loglik=True)
            return np.sum(ml), np.sum(nl)

        red = gpflow.utilities.deepcopy(m)
        red.kernel.kernels.pop(j)
        red.num_trainable_params = np.nan
        red.optimize_params(data=data,
                            optimizer=getattr(m, "optimizer", None) or "scipy")

        e_f, e_r = elbo(m), elbo(red)
        p_f, nul = pred_ll(m)
        p_r, _ = pred_ll(red)
        rows.append({
            "group": label, "metabolite": met, "covariate": cov,
            "stored_DE": de, "stored_log_bf": log_bf,
            "kernel_variance": var, "at_variance_floor": var <= VARIANCE_FLOOR * 10,
            "lengthscale": ls, "contribution_sd": float(tgt.std()),
            "r2_on_other_components": r2,
            "d_elbo": e_f - e_r,
            "d_pred_deviance": -2.0 * (p_r - p_f),
            "pred_gain_over_null": -2.0 * (nul - p_f),
        })
        if i % 10 == 0:
            print(f"  {i}/{len(picks)} ({(time.time()-t0)/60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}\n")
    pd.set_option("display.width", 200)
    print(df.groupby("group")[[
        "kernel_variance", "contribution_sd", "r2_on_other_components",
        "d_elbo", "d_pred_deviance"]].median().to_string())
    print("\nat the variance floor, by group:")
    print(df.groupby("group").at_variance_floor.mean().to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
