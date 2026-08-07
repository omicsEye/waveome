"""Sanity-check the within-between (Mundlak) decomposition of hbi.

Replaces the single `hbi` covariate with two:
    hbi_between_ij = mean of hbi over subject i   (constant within subject)
    hbi_within_ij  = hbi_ij - hbi_between_ij      (deviation, sums to 0)

and refits a handful of metabolites. The question this answers: does the
signal that currently loads onto SE[hbi] / lin[hbi] split sensibly, and in
particular does HILp_QI2874 -- whose SE[hbi] survived a within-unit
permutation at log_bf ~36, suggesting it was proxying subject structure --
move its signal onto the *between* term?

Also reports the conditioning of the decomposition (the two new columns must
not be collinear with each other or with the other covariates, or the
collinearity guard will halt and the split won't be identifiable).
"""
import pickle

import gpflow
import numpy as np


from waveome.kernels import Lin
from waveome.model_search import GPSearch

INPUT_FP = (
    "output/fit_penalized_models_revision_full_scipy_ls_prior_no_prune_clamp_removed.pkl"
)
DEMO = [
    "HILp_QI2874",   # SE:hbi 36.6 -- suspected subject-structure proxy
    "C8p_QI92",      # SE:hbi 6.1
    "HILn_QI112",    # lin:hbi 25.3
    "C18n_QI41",     # lin:hbi 18.0
    "C8p_QI140",     # null
    "HILp_QI10330",  # null
]


def main():
    with open(INPUT_FP, "rb") as f:
        gps = pickle.load(f)

    X = gps.X.copy()
    unit = X["participant_id"].values
    hbi = X["hbi"].values

    between = np.array([hbi[unit == u].mean() for u in unit])
    within = hbi - between

    X_dec = X.drop(columns=["hbi"]).copy()
    X_dec["hbi_between"] = between
    X_dec["hbi_within"] = within

    # --- conditioning diagnostics ---
    cont = ["hbi_between", "hbi_within", "study_days", "age", "time_from_max"]
    print("pairwise |r| among continuous covariates after decomposition:")
    worst = 0.0
    for i in range(len(cont)):
        for j in range(i + 1, len(cont)):
            r = np.corrcoef(X_dec[cont[i]], X_dec[cont[j]])[0, 1]
            flag = "  <-- HALT" if abs(r) > 0.95 else ("  <- warn" if abs(r) > 0.8 else "")
            worst = max(worst, abs(r))
            print(f"  {cont[i]:14s} vs {cont[j]:14s} r={r:+.3f}{flag}")
    print(f"worst |r| = {worst:.3f} (halt cutoff 0.95, warn 0.80)\n")

    var_b = between.var()
    var_w = within.var()
    print(f"variance split: between={var_b:.3f} ({100*var_b/(var_b+var_w):.0f}%)  "
          f"within={var_w:.3f} ({100*var_w/(var_b+var_w):.0f}%)\n")

    # --- refit the decomposed model on the demo metabolites ---
    gps_dec = GPSearch(
        X=X_dec,
        Y=gps.Y[DEMO].copy(),
        unit_col="participant_id",
        categorical_vars=["site_name", "race", "sex", "general_wellbeing"],
        outcome_likelihood="negativebinomial",
        Y_transform=None,
    )
    gps_dec.penalized_optimization(
        random_seed=9102,
        kernel_options={
            "second_order_numeric": False,
            "unit_numeric_interactions": False,
            "categorical_numeric_interactions": False,
            "kerns": [gpflow.kernels.SquaredExponential(), Lin()],
        },
        num_restart=3,
        optimization_options={"optimizer": "scipy"},
        prune_components=False,
    )

    tab = gps_dec.get_significance_table()
    tab.to_csv("output/decomposition_sanity_check.csv", index=False)

    print("\n=== log_bf: original single hbi  vs  decomposed between/within ===")
    print(f"{'metabolite':14s}{'orig SE':>9s}{'orig lin':>9s} | "
          f"{'SE_btwn':>9s}{'lin_btwn':>9s}{'SE_wthn':>9s}{'lin_wthn':>9s}")
    for name in DEMO:
        od = gps.models[name].feature_importance_detail
        sub = tab[tab["metabolite"] == name]

        def g(kt, cov):
            r = sub[(sub["kernel_type"] == kt) & (sub["covariate"] == cov)]
            return r["log_bf"].iloc[0] if len(r) else np.nan

        print(f"{name:14s}{od[5]['log_bf']:9.1f}{od[6]['log_bf']:9.1f} | "
              f"{g('squared_exponential','hbi_between'):9.1f}"
              f"{g('lin','hbi_between'):9.1f}"
              f"{g('squared_exponential','hbi_within'):9.1f}"
              f"{g('lin','hbi_within'):9.1f}")

    print("\nparticipant_id log_bf (orig vs decomposed) -- does the subject term "
          "change once between-subject hbi has its own home?")
    for name in DEMO:
        od = gps.models[name].feature_importance_detail
        sub = tab[(tab["metabolite"] == name)
                  & (tab["covariate"] == "participant_id")]
        v = sub["log_bf"].iloc[0] if len(sub) else np.nan
        print(f"  {name:14s} orig={od[0]['log_bf']:8.1f}   decomposed={v:8.1f}")

    print("\nwrote output/decomposition_sanity_check.csv")


if __name__ == "__main__":
    main()
