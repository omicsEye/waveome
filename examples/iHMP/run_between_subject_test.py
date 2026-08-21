"""Between-subject significance test, the complement to the within-subject one.

The within-subject permutation deliberately preserves each subject's own set
of covariate values, so between-subject association sits in the null on both
sides and can neither create nor earn a hit. That leaves 41% of hbi's variance
and 47% of time_from_max's untested. This covers it.

Procedure, per (metabolite, covariate):
  1. refit the model with BOTH of that covariate's kernel components dropped,
     so the residuals hold whatever the rest of the model cannot explain --
     including any covariate signal, which is the point
  2. reduce each subject to two numbers: mean Pearson residual, mean covariate
  3. Spearman correlation across the 49 subjects (Spearman, not Pearson: hbi
     is a bounded discrete clinical score, skew +2.03, 14 distinct values)
  4. permute the 49 subject-level covariate means and recompute, B times
  5. empirical p, then BH across metabolites within each covariate

Why this is exact and needs no matched block sizes: each subject contributes
exactly ONE number, so subjects are freely exchangeable under H0 regardless of
how many visits each had. It is the same move OmicsLonDA makes with a
subject-level group label, applied to a continuous value.

Cost is one extra fit per (metabolite, covariate) -- there are no refits under
permutation, since step 4 is arithmetic on 49 values.
"""
import os
import pickle
import time

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gpflow  # noqa: E402
import ray  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from waveome.model_search import _component_covariate_names  # noqa: E402
from waveome.utilities import (  # noqa: E402
    calc_bh_qvalues,
    calc_residuals,
    convert_data_to_tensors,
)

IN = "output/fit_penalized_models_revision_full_scipy_ls_prior_no_prune_clamp_removed.pkl"
OUT = "output/ihmp_between_subject.csv"
SEED = 9102
B = 20000          # >= m/q = 11,280 for m=564 at q=0.05
COVARIATES = ["hbi", "time_from_max"]


@ray.remote(max_calls=1, max_retries=5)
def subject_residuals(model, X, y, drop_idx):
    """Refit without the covariate's components; return per-observation resids."""
    try:
        data = convert_data_to_tensors(X, y.reshape(-1, 1))
        m = gpflow.utilities.deepcopy(model)
        for i in sorted(drop_idx, reverse=True):
            m.kernel.kernels.pop(i)
        m.num_trainable_params = np.nan
        m.optimize_params(data=data,
                          optimizer=getattr(model, "optimizer", None) or "scipy")
        return calc_residuals(m, X=data[0], Y=data[1],
                              resid_type="pearson").ravel(), None
    except Exception as e:
        return None, str(e)


def main():
    with open(IN, "rb") as f:
        gps = pickle.load(f)
    X = gps.X.to_numpy()
    unit = X[:, gps.unit_idx]
    units = np.unique(unit)
    names = list(gps.models.keys())
    kts, cns = _component_covariate_names(
        gps.models[names[0]].kernel_name, gps.feat_names)
    print(f"{len(names)} metabolites, {len(units)} subjects")

    ray.init(include_dashboard=False, configure_logging=False)
    X_ref = ray.put(X)
    jobs = []
    for cov in COVARIATES:
        drop = [i for i, c in enumerate(cns) if c == cov]
        print(f"{cov}: dropping components {drop} "
              f"({[kts[i] for i in drop]}) to build residuals")
        for n in names:
            jobs.append((n, cov, subject_residuals.remote(
                ray.put(gps.models[n]), X_ref,
                ray.put(gps.Y[n].to_numpy()), drop)))

    print(f"\n{len(jobs)} refits queued (one per metabolite x covariate)")
    t0 = time.time()
    resid, n_fail, done = {}, 0, 0
    pending = [j[2] for j in jobs]
    meta = {j[2]: j[:2] for j in jobs}
    while pending:
        ready, pending = ray.wait(pending, num_returns=min(100, len(pending)))
        for ref in ready:
            key = meta[ref]
            r, err = ray.get(ref)
            done += 1
            if err is None:
                resid[key] = r
            else:
                n_fail += 1
        el = time.time() - t0
        print(f"  {done}/{len(jobs)} ({el/60:.1f} min, {n_fail} failed)",
              flush=True)
    ray.shutdown()
    np.savez_compressed(
        "output/ihmp_between_subject_residuals.npz",
        **{f"{n}|{c}": v for (n, c), v in resid.items()})
    print("cached residuals -> output/ihmp_between_subject_residuals.npz")

    rng = np.random.default_rng(SEED)
    rows = []
    for cov in COVARIATES:
        ci = gps.feat_names.index(cov)
        cov_mean = np.array([X[unit == u, ci].mean() for u in units])
        # One permutation set shared across metabolites: the covariate means
        # are the same vector for every metabolite, so the null is too.
        # B must satisfy N >= m/q or BH cannot reject at rank 1: at m=564,
        # q=0.05 that is 11,280. An earlier version used 200 and floored every
        # p-value at 5e-03, 56x too coarse, reporting a meaningless 0/564.
        rc = rankdata(cov_mean)
        rc = (rc - rc.mean()) / np.linalg.norm(rc - rc.mean())
        perms = np.array([rng.permutation(rc) for _ in range(B)])   # (B, n_units)
        for n in names:
            r = resid.get((n, cov))
            if r is None:
                continue
            y_mean = np.array([r[unit == u].mean() for u in units])
            ry = rankdata(y_mean)
            ry = (ry - ry.mean()) / np.linalg.norm(ry - ry.mean())
            obs = abs(float(rc @ ry))
            null = np.abs(perms @ ry)          # all B draws in one matmul
            p = (1 + np.sum(null >= obs)) / (1 + len(null))
            rows.append({"metabolite": n, "covariate": cov,
                         "spearman_abs": obs, "p_value": p})
    df = pd.DataFrame(rows)
    for cov, g in df.groupby("covariate"):
        df.loc[g.index, "q_value"] = calc_bh_qvalues(g["p_value"].to_numpy())
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print("\n=== between-subject, significant per covariate ===")
    for cov, g in df.groupby("covariate"):
        print(f"  {cov:16s} q<0.05: {int((g.q_value < 0.05).sum()):3d}/{len(g)}"
              f"   q<0.10: {int((g.q_value < 0.10).sum()):3d}/{len(g)}"
              f"   min p {g.p_value.min():.3g}")


if __name__ == "__main__":
    main()
