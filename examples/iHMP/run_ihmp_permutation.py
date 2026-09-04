"""Permutation significance for the reported iHMP strata.

Uses the fitted no-prune models as-is -- no refitting of the models
themselves, only the drop-one refits each permutation draw requires.

B1 is not set here. The library derives it per covariate after the B0
screen, as (m / n_live) / q_target clipped to [B1_min, B1_max], because the
requirement depends on how many components survive the screen -- which is
only knowable once the screen has run.

At q_target=0.05 that gives 100 for both covariates in this cohort. The
earlier hardcoded B1=60 was justified on resolution grounds alone and it did
clear BH's rank-1 threshold for hbi; what it did not cover is the stability
of each component's OWN null SD, which is a separate failure mode. At B=60
one time_from_max hit had q=0.009 and moved to q=1.00 once its null was
estimated from 120 draws (FINDINGS 27).
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
    # B1 is DERIVED, not fixed: the library computes
    #   B1 >= (m / n_live) / q_target
    # per covariate after the B0 screen, clipped to [B1_min, B1_max].
    # At q_target=0.05 this gives 100 for both covariates here.
    # Hardcoding B1=60 previously left hbi below the library's own
    # default and time_from_max needing a separate manual top-up.
    B0=10, q_target=0.05, random_seed=9102, verbose=True,
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
