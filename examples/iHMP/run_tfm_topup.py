"""Top up time_from_max only, to B1=120.

lin:time_from_max cleared BH's rank-1 threshold by just 3% (floor 8.59e-05 vs
8.87e-05 needed) with all 3 of its hits tied at that floor -- the one stratum
where more draws could plausibly change the count. This raises its pooled
draws to ~18,840, a 1.67x margin.

hbi is deliberately left at B1=60: it already has 1.7x margin, and topping it
up would cost ~11h more for no change in its conclusions.

Resumes from the shared checkpoint, so only draws 60-119 are computed.
"""
import pickle, time
import pandas as pd

IN = "output/fit_penalized_models_revision_full_scipy_ls_prior_no_prune_clamp_removed.pkl"
DRAWS = "output/ihmp_permutation_draws.csv"

with open(IN, "rb") as f:
    gps = pickle.load(f)
t0 = time.time()
res = gps.permutation_significance(
    covariates=["time_from_max"], B0=10, B1=120, random_seed=9102,
    checkpoint_path=DRAWS, resume=True, verbose=True,
)
print(f"\ntop-up took {(time.time()-t0)/3600:.1f} h")
res.to_csv("output/ihmp_permutation_tfm_b120.csv", index=False)
print("wrote output/ihmp_permutation_tfm_b120.csv")
for s, g in res.groupby("stratum"):
    fl = g.p_value.min()
    print(f"  {s:36s} q<0.05: {int((g.q_value<0.05).sum()):3d}   "
          f"q<0.10: {int((g.q_value<0.10).sum()):3d}   min p {fl:.2e}   "
          f"tied at min: {int((g.p_value<=fl*1.001).sum())}")
