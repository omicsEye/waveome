"""Between-subject test, current vs. proposed refit (internal review issue 4).

Current: the refit drops only the covariate's own components, keeping the
participant kernel. Proposed: it also drops every component involving the
participant column. Both run on the same fitted models, seed and draws.
"""
import os
import pickle
import sys

import pandas as pd

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "output"
IN = os.path.join(OUT_DIR, "ihmp_penalized_fit.pkl")
COVS = ["hbi", "time_from_max"]

with open(IN, "rb") as f:
    gps = pickle.load(f)
print(f"{len(gps.models)} metabolites, {gps.X.shape[0]} observations")

res = {}
for label, flag in [("current", False), ("proposed", True)]:
    print(f"\n===== {label} (drop_unit_components={flag}) =====", flush=True)
    r = gps.between_subject_significance(
        covariates=COVS, n_permutations=20000, random_seed=9102,
        drop_unit_components=flag, verbose=True)
    r.to_csv(os.path.join(OUT_DIR, f"ihmp_between_subject_issue4_{label}.csv"),
             index=False)
    res[label] = r

print("\n=== significant per covariate ===")
for label, r in res.items():
    for cov, g in r.groupby("covariate"):
        print(f"  {label:9s} {cov:14s} q<0.05: {(g.q_value < 0.05).sum():3d}"
              f"  q<0.10: {(g.q_value < 0.10).sum():3d}  of {len(g)}"
              f"  min p {g.p_value.min():.2g}  median |rho| "
              f"{g.spearman_abs.median():.3f}")

m = res["current"].merge(res["proposed"], on=["metabolite", "covariate"],
                         suffixes=("_cur", "_new"))
print("\n=== per-metabolite agreement ===")
for cov, g in m.groupby("covariate"):
    rho = g.spearman_abs_cur.corr(g.spearman_abs_new, method="spearman")
    print(f"  {cov}: rank corr of |rho| current vs proposed {rho:.2f}; "
          f"|rho| larger under proposed for {(g.spearman_abs_new > g.spearman_abs_cur).mean():.0%}")
print("\nsmallest q under proposed:")
print(m.nsmallest(12, "q_value_new")[
    ["metabolite", "covariate", "spearman_abs_cur", "q_value_cur",
     "spearman_abs_new", "p_value_new", "q_value_new"]].to_string(index=False))
