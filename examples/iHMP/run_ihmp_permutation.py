"""Permutation significance for the reported iHMP strata.

Uses the fitted no-prune models as-is -- no refitting of the models
themselves, only the drop-one refits each permutation draw requires.

B1=60 rather than 100: p-values come from quantile regression pooled over
every draw, so the resolution floor is 1/N_total (~2.4e-5 here), comfortably
below BH's rank-1 threshold of 0.05/564 = 8.9e-5. Under the earlier
scale-binned scheme each bin was its own pool and 100 was needed; that is no
longer the binding constraint.
"""
import pickle
import time

import pandas as pd

IN = ("output/fit_penalized_models_revision_full_scipy_ls_prior"
      "_no_prune_clamp_removed.pkl")
OUT = "output/ihmp_permutation_significance.csv"
DRAWS = "output/ihmp_permutation_draws.csv"

with open(IN, "rb") as f:
    gps = pickle.load(f)
print(f"{len(gps.models)} metabolites, {gps.X.shape[0]} observations")

t0 = time.time()
res = gps.permutation_significance(
    covariates=["hbi", "time_from_max"],
    B0=10, B1=60, random_seed=9102, verbose=True,
    # Checkpoint every batch: this run is ~29 h, and an earlier attempt that
    # persisted only at the end lost ~21 h when it was interrupted. Re-running
    # this same command picks up wherever it stopped.
    checkpoint_path=DRAWS, resume=True,
)
print(f"\ntotal {(time.time()-t0)/3600:.1f} h")

res.to_csv(OUT, index=False)
gps.permutation_draws.to_csv(DRAWS, index=False)
print(f"wrote {OUT} and {DRAWS}")

print("\n=== significant at q<0.05, per stratum ===")
for s, g in res.groupby("stratum"):
    n = int((g.q_value < 0.05).sum())
    print(f"  {s:36s} {n:3d}/{len(g)}")
print("\ntop hits:")
top = res.nsmallest(12, "q_value")
print(top[["metabolite", "stratum", "log_bf", "null_centre", "null_sd",
           "p_value", "q_value"]].to_string(index=False))
