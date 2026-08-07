"""Within-subject permutation significance for time-varying covariates.

For each (metabolite, covariate) the covariate's values are freely shuffled
INSIDE each subject, leaving every subject's own multiset -- and therefore
all between-subject structure -- exactly intact. The drop-one log_bf is then
recomputed for every kernel component of that covariate. Because the observed
statistic is recomputed through the identical code path (a zero shuffle)
rather than read from the stored fit, optimizer/ELBO idiosyncrasies appear on
both sides and cancel in the ranking.

Tests: "does this metabolite track the covariate as a subject's own value
changes?" Between-subject association stays in the null on both sides, so it
can neither create nor be credited as a hit.

p-values use per-metabolite centering with a pooled tail shape. B draws per
metabolite is far too coarse for a per-metabolite p-value (min p = 1/(B+1)),
but it locates each metabolite's null centre precisely enough; the centred
draws are then pooled across metabolites within a (kernel, covariate)
stratum to give a shared, fine-grained tail. The homogeneity that pooling
assumes is reported as a diagnostic rather than taken on faith.

Raw draws are always saved so diagnostics never require a re-run.

    python within_subject_permutation.py --B 20
    python within_subject_permutation.py --B 10 --n-metabolites 10   # smoke test
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

# Pin per-worker threads BEFORE tensorflow is imported: each fit is small, so
# intra-op parallelism just makes Ray workers fight over the same cores.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, "1")

import gpflow  # noqa: E402
import ray  # noqa: E402
from gpflow.utilities import set_trainable  # noqa: E402

from waveome.model_search import _component_covariate_names  # noqa: E402
from waveome.utilities import (  # noqa: E402
    calc_bh_qvalues,
    convert_data_to_tensors,
)

INPUT_FP = (
    "output/fit_penalized_models_revision_full_scipy_ls_prior_no_prune_clamp_removed.pkl"
)
SEED = 9102
UNIT_IDX = 0
# log_bf values that differ only at optimizer-noise level are the SAME value;
# without this, a null atom straddling the observed flips p wildly (observed
# -0.9681 vs -0.9675 gave p=0.90 and p=0.31 for what is one collapsed state).
TIE_TOL = 1e-3


def shuffle_within(X, col, rng):
    out = X.copy()
    for u in np.unique(X[:, UNIT_IDX]):
        i = np.where(X[:, UNIT_IDX] == u)[0]
        out[i, col] = rng.permutation(X[i, col])
    return out


def _bic(m, data):
    a, b = m.q_mu.trainable, m.q_sqrt.trainable
    set_trainable(m.q_mu, True)
    set_trainable(m.q_sqrt, True)
    try:
        return m.calc_metric(data=data, metric="BIC")
    finally:
        set_trainable(m.q_mu, a)
        set_trainable(m.q_sqrt, b)


@ray.remote(max_calls=1, max_retries=5)
def draw(model, X, y, col, targets, seed):
    """One draw. seed=None -> observed (no shuffle). Returns {kernel_idx: log_bf}."""
    try:
        Xp = X if seed is None else shuffle_within(
            X, col, np.random.default_rng(seed))
        data = convert_data_to_tensors(Xp, y.reshape(-1, 1))
        opt = getattr(model, "optimizer", None) or "scipy"
        full = gpflow.utilities.deepcopy(model)
        full.num_trainable_params = np.nan
        full.optimize_params(data=data, optimizer=opt)
        bf = _bic(full, data)
        out = {}
        for idx in targets:
            red = gpflow.utilities.deepcopy(full)
            red.kernel.kernels.pop(idx)
            red.num_trainable_params = np.nan
            red.optimize_params(data=data, optimizer=opt)
            out[idx] = float(-0.5 * (bf - _bic(red, data)))
        return out, None
    except Exception as e:  # keep one bad component from killing the batch
        return None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=20)
    ap.add_argument("--covariates", nargs="+", default=["hbi", "time_from_max"])
    ap.add_argument("--n-metabolites", type=int, default=None)
    ap.add_argument("--metabolites", nargs="+", default=None,
                    help="explicit metabolite names (overrides --n-metabolites)")
    ap.add_argument("--out-prefix", default="output/within_perm")
    args = ap.parse_args()

    with open(INPUT_FP, "rb") as f:
        gps = pickle.load(f)
    X = gps.X.to_numpy()
    feat = gps.feat_names

    names = list(gps.models.keys())
    if args.metabolites:
        names = list(args.metabolites)
    elif args.n_metabolites:
        names = names[: args.n_metabolites]

    # Structure is identical across metabolites, so resolve component indices once.
    kts, cns = _component_covariate_names(
        gps.models[names[0]].kernel_name, feat)
    cov_targets = {
        c: [i for i, cn in enumerate(cns) if cn == c] for c in args.covariates
    }
    for c, idxs in cov_targets.items():
        print(f"{c}: kernel components {idxs} -> {[kts[i] for i in idxs]}")
        if not idxs:
            raise ValueError(f"covariate {c!r} has no kernel components")

    ray.init(include_dashboard=False, configure_logging=False)
    X_ref = ray.put(X)

    jobs = []
    for c in args.covariates:
        col = feat.index(c)
        tgt = cov_targets[c]
        for mi, name in enumerate(names):
            m_ref = ray.put(gps.models[name])
            y_ref = ray.put(gps.Y[name].to_numpy())
            jobs.append((name, c, -1,
                         draw.remote(m_ref, X_ref, y_ref, col, tgt, None)))
            for b in range(args.B):
                jobs.append((name, c, b, draw.remote(
                    m_ref, X_ref, y_ref, col, tgt,
                    SEED + 100003 * mi + 7919 * args.covariates.index(c) + b)))

    print(f"\n{len(jobs)} draws queued "
          f"({len(names)} metabolites x {len(args.covariates)} covariates "
          f"x (1 observed + {args.B} permutations))")
    t0 = time.time()

    rows, n_fail = [], 0
    pending = [j[3] for j in jobs]
    meta = {j[3]: j[:3] for j in jobs}
    done_n = 0
    while pending:
        done, pending = ray.wait(pending, num_returns=min(100, len(pending)))
        for ref in done:
            name, cov, b = meta[ref]
            res, err = ray.get(ref)
            done_n += 1
            if err is not None:
                n_fail += 1
                continue
            for idx, v in res.items():
                rows.append({"metabolite": name, "covariate": cov,
                             "kernel_type": kts[idx], "kernel_idx": idx,
                             "draw": b, "log_bf": v})
        el = time.time() - t0
        print(f"  {done_n}/{len(jobs)} draws ({el/60:.1f} min, "
              f"{el/done_n:.2f} s/draw, {n_fail} failed)", flush=True)

    draws = pd.DataFrame(rows)
    draws.to_csv(f"{args.out_prefix}_raw_draws.csv", index=False)
    print(f"\nwrote {args.out_prefix}_raw_draws.csv ({len(draws)} rows, "
          f"{n_fail} failed draws)")

    # ---- per-metabolite centring, pooled tail shape, BH per stratum ----
    out = []
    for (cov, kt), grp in draws.groupby(["covariate", "kernel_type"]):
        obs = grp[grp.draw == -1].set_index("metabolite")["log_bf"]
        null = grp[grp.draw >= 0]
        centre = null.groupby("metabolite")["log_bf"].median()
        spread = null.groupby("metabolite")["log_bf"].std()
        pooled = (null["log_bf"].values
                  - centre.reindex(null["metabolite"]).values)

        common = obs.index.intersection(centre.index)
        excess = obs.loc[common].values - centre.loc[common].values
        p = np.array([(1 + np.sum(pooled >= e - TIE_TOL)) / (1 + len(pooled))
                      for e in excess])
        q = calc_bh_qvalues(p)
        for m, o, c_, e, pv, qv in zip(common, obs.loc[common], centre.loc[common],
                                       excess, p, q):
            out.append({"metabolite": m, "covariate": cov, "kernel_type": kt,
                        "log_bf": o, "null_centre": c_, "excess": e,
                        "p_within": pv, "q_within": qv})
        print(f"\n{kt}:{cov}  pooled null n={len(pooled)} "
              f"(resolution {1/(1+len(pooled)):.2g})")
        print(f"  per-metabolite null spread: median={spread.median():.3f} "
              f"IQR=[{spread.quantile(.25):.3f}, {spread.quantile(.75):.3f}] "
              f"max={spread.max():.3f}   <- pooling assumes these are comparable")
        print(f"  significant at q<0.05: {int(np.sum(q < 0.05))}/{len(q)}")

    res = pd.DataFrame(out).sort_values(["covariate", "kernel_type", "q_within"])
    res.to_csv(f"{args.out_prefix}_significance.csv", index=False)
    print(f"\nwrote {args.out_prefix}_significance.csv")
    ray.shutdown()


if __name__ == "__main__":
    main()
