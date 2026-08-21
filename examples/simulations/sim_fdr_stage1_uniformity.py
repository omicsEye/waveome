"""T2 Stage 1 -- are the pooled permutation p-values calibrated?

Simulates the null the within-subject test actually targets: `cindex` has
NO within-subject effect on the outcome, but MAY have a real between-subject
one. Valid p-values must then be uniform on (0,1). If pooling a shared tail
shape across outcomes is anti-conservative -- the worry raised by the iHMP
mixed panel, where per-outcome null SDs ranged 0.00 to 18.43 -- the p-values
will be stochastically too small and the QQ plot against uniform will bend.

Getting that null right is the whole point. A first version simulated a
COMPLETE null (no cindex effect at all, within or between) and was vacuous:
the wide, heterogeneous nulls arise precisely BECAUSE a within-subject
shuffle preserves the between-subject association, so an SE kernel keeps
fitting it. Remove that association from the data and every null collapses
to a point mass no matter how the subject effect is tuned -- observed
across two tunings, categorical[id] collapsing in 83% and then 10% of
outcomes, with SE[cindex] degenerate either way. The complete null is also
a stricter hypothesis than the one under test: H0-within permits a
between-subject effect.

The simulated data deliberately reproduces the structural features that
produced the problem, because a naive simulation passes trivially:
  * unequal visits per subject (Poisson, as in the real cohort)
  * `cindex`: a covariate carrying BOTH within- and between-subject
    variance, targeted at the ~41% between-subject split hbi shows. The
    existing sim generator has only `time` (within) and `treat` (between),
    neither of which exercises this.
  * real subject-level structure in the outcome, so an SE kernel on a
    clustered covariate has something to proxy
  * negative-binomial likelihood + horseshoe, matching the real pipeline,
    so components collapse to the variance floor the same way
  * both SE and lin components on the tested covariate (the proxying
    problem was SE-specific and absent for lin)

    python sim_fdr_stage1_uniformity.py --M 150 --B 20
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, "1")

import gpflow  # noqa: E402
import ray  # noqa: E402
from gpflow.utilities import set_trainable  # noqa: E402

from waveome.kernels import Lin  # noqa: E402
from waveome.model_search import GPSearch, _component_covariate_names  # noqa: E402
from waveome.utilities import calc_bh_qvalues, convert_data_to_tensors  # noqa: E402

SEED = 9102
UNIT_IDX = 0
TIE_TOL = 1e-3
TARGET_BETWEEN_FRAC = 0.41   # matches hbi in the real cohort


def simulate(n_units=49, visit_rate=5, M=150, rng=None):
    """No WITHIN-subject cindex effect; a between-subject one may be present."""
    rng = rng or np.random.default_rng(SEED)
    n_obs = np.maximum(rng.poisson(visit_rate, n_units), 1)

    unit = np.repeat(np.arange(n_units), n_obs)
    time = np.concatenate([np.sort(rng.uniform(-2, 2, k)) for k in n_obs])

    # cindex = between-subject level + within-subject deviation.
    # sd_b/sd_w chosen so var_between/(var_between+var_within) ~ TARGET.
    sd_w = 1.0
    sd_b = sd_w * np.sqrt(TARGET_BETWEEN_FRAC / (1 - TARGET_BETWEEN_FRAC))
    cindex = (np.repeat(rng.normal(0, sd_b, n_units), n_obs)
              + rng.normal(0, sd_w, len(unit)))

    X = pd.DataFrame({"id": unit.astype(float),
                      "cindex": cindex,
                      "time": time})

    # Outcome: subject offset + smooth time effect + an optional BETWEEN-
    # subject cindex effect. Never a within-subject cindex effect, so the
    # hypothesis under test is exactly true for every outcome.
    #
    # Subject-effect strength is also drawn per outcome, since real
    # metabolites span a range (participant_id collapsed in 52% of them) and
    # a single fixed level puts every outcome in the same regime.
    #
    # Subject-mean of cindex: the ONLY channel through which cindex may act.
    # It is constant within subject, so it survives a within-subject shuffle
    # untouched -- which is what lets an SE kernel keep fitting it under the
    # null and produces the wide, heterogeneous nulls we need to stress.
    c_between = np.repeat(
        [cindex[unit == u].mean() for u in np.arange(n_units)], n_obs)

    Y, subj_sd, betas = {}, {}, {}
    for m in range(M):
        sd_u = np.exp(rng.uniform(np.log(0.1), np.log(1.0)))
        u = np.repeat(rng.normal(0, sd_u, n_units), n_obs)
        amp, freq = rng.uniform(0.3, 1.0), rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 6.3)
        # Between-subject cindex effect: zero for ~30% of outcomes, otherwise
        # spanning weak to strong. There is NEVER a within-subject cindex
        # effect, so H0-within holds exactly for every outcome and all
        # p-values must be uniform.
        beta = 0.0 if rng.random() < 0.3 else np.exp(
            rng.uniform(np.log(0.2), np.log(1.5)))
        # Baseline abundance spanning the real cohort's per-metabolite medians
        # (q10 5.9e3 to q90 1.2e7). A first pass used a fixed intercept of 1.5,
        # i.e. ~5 counts -- five orders of magnitude low, so Poisson noise
        # swamped everything and every component collapsed.
        intercept = rng.uniform(np.log(5.9e3), np.log(1.2e7))
        eta = (intercept + u + amp * np.sin(freq * time + phase)
               + beta * c_between)
        mu = np.exp(eta)
        # NB via gamma-Poisson mixture, dispersion r (higher = less noise;
        # r=5 previously buried the subject effect under sampling noise)
        r = 20.0
        Y[f"y{m}"] = rng.poisson(rng.gamma(r, mu / r))
        subj_sd[f"y{m}"] = sd_u
        betas[f"y{m}"] = beta
    return X, pd.DataFrame(Y), subj_sd, betas


def between_fraction(x, unit):
    gm = x.mean()
    us = np.unique(unit)
    b = sum(np.sum(unit == u) * (x[unit == u].mean() - gm) ** 2 for u in us)
    w = sum(np.sum((x[unit == u] - x[unit == u].mean()) ** 2) for u in us)
    return b / (b + w)


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
    except Exception as e:
        return None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=150)
    ap.add_argument("--B", type=int, default=20)
    ap.add_argument("--n-units", type=int, default=49)
    ap.add_argument("--out-prefix", default="sim_waveome_output/t2_stage1")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    X, Y, subj_sd, betas = simulate(n_units=args.n_units, M=args.M, rng=rng)
    print(f"simulated: {X.shape[0]} obs, {args.n_units} subjects, {args.M} outcomes")
    print(f"visits per subject: min={int(X.groupby('id').size().min())} "
          f"median={int(X.groupby('id').size().median())} "
          f"max={int(X.groupby('id').size().max())}")
    bfrac = between_fraction(X['cindex'].values, X['id'].values)
    print(f"cindex between-subject fraction: {bfrac:.3f} "
          f"(target {TARGET_BETWEEN_FRAC}, real hbi 0.413)")
    nb = sum(1 for b in betas.values() if b > 0)
    print(f"outcomes with a real BETWEEN-subject cindex effect: {nb}/{args.M}")
    print("no outcome has a WITHIN-subject cindex effect -> all "
          "p-values should be U(0,1)\n")

    gps = GPSearch(X=X, Y=Y, unit_col="id", categorical_vars=[],
                   outcome_likelihood="negativebinomial", Y_transform=None)
    t0 = time.time()
    gps.penalized_optimization(
        random_seed=SEED,
        kernel_options={"second_order_numeric": False,
                        "unit_numeric_interactions": False,
                        "categorical_numeric_interactions": False,
                        "kerns": [gpflow.kernels.SquaredExponential(), Lin()]},
        num_restart=3,
        optimization_options={"optimizer": "scipy"},
        prune_components=False,
    )
    print(f"fitting done in {(time.time()-t0)/60:.1f} min")

    names = list(gps.models.keys())
    kts, cns = _component_covariate_names(gps.models[names[0]].kernel_name,
                                          gps.feat_names)

    # GATE: does the simulation actually reproduce the regime that motivated
    # this check? If subject structure is uniformly shrunk away, SE[cindex]
    # never proxies it, every null is degenerate, and the calibration result
    # is vacuous -- it can only pass. Verify before reading any p-value.
    print("\n--- regime gate (must hold or the check is uninformative) ---")
    for i, (kt, cn) in enumerate(zip(kts, cns)):
        v = np.array([float(m.kernel.kernels[i].variance.numpy())
                      for m in gps.models.values()])
        print(f"  {kt+'['+cn+']':30s} frac collapsed = {np.mean(v < 1e-8):.2f}")
    print("  target: categorical[id] ~0.5 (real cohort 0.52); a first pass at "
          "0.83 was too shrunk to be informative")
    targets = [i for i, c in enumerate(cns) if c == "cindex"]
    print(f"cindex components: {targets} -> {[kts[i] for i in targets]}")
    col = gps.feat_names.index("cindex")

    Xn = gps.X.to_numpy()
    X_ref = ray.put(Xn)
    jobs = []
    for mi, name in enumerate(names):
        m_ref = ray.put(gps.models[name])
        y_ref = ray.put(gps.Y[name].to_numpy())
        jobs.append((name, -1, draw.remote(m_ref, X_ref, y_ref, col, targets, None)))
        for b in range(args.B):
            jobs.append((name, b, draw.remote(m_ref, X_ref, y_ref, col, targets,
                                              SEED + 100003 * mi + b)))

    print(f"\n{len(jobs)} draws queued")
    t0 = time.time()
    rows, n_fail, done_n = [], 0, 0
    pending = [j[2] for j in jobs]
    meta = {j[2]: j[:2] for j in jobs}
    while pending:
        done, pending = ray.wait(pending, num_returns=min(100, len(pending)))
        for ref in done:
            name, b = meta[ref]
            res, err = ray.get(ref)
            done_n += 1
            if err is not None:
                n_fail += 1
                continue
            for idx, v in res.items():
                rows.append({"outcome": name, "kernel_type": kts[idx],
                             "draw": b, "log_bf": v})
        el = time.time() - t0
        print(f"  {done_n}/{len(jobs)} ({el/60:.1f} min, {el/done_n:.2f} s/draw, "
              f"{n_fail} failed)", flush=True)

    draws = pd.DataFrame(rows)
    draws.to_csv(f"{args.out_prefix}_raw_draws.csv", index=False)

    from scipy.stats import kstest
    out = []
    for kt, grp in draws.groupby("kernel_type"):
        obs = grp[grp.draw == -1].set_index("outcome")["log_bf"]
        null = grp[grp.draw >= 0]
        centre = null.groupby("outcome")["log_bf"].median()
        spread = null.groupby("outcome")["log_bf"].std()
        pooled = null["log_bf"].values - centre.reindex(null["outcome"]).values
        common = obs.index.intersection(centre.index)
        excess = obs.loc[common].values - centre.loc[common].values
        p = np.array([(1 + np.sum(pooled >= e - TIE_TOL)) / (1 + len(pooled))
                      for e in excess])
        q = calc_bh_qvalues(p)
        ks = kstest(p, "uniform")
        print(f"\n=== {kt}:cindex (COMPLETE NULL) ===")
        print(f"  per-outcome null SD: median={spread.median():.3f} "
              f"IQR=[{spread.quantile(.25):.3f}, {spread.quantile(.75):.3f}] "
              f"max={spread.max():.3f}")
        print(f"  null-scale heterogeneity: {int((spread > 0.5).sum())}/"
              f"{len(spread)} outcomes with SD>0.5, "
              f"{int((spread < 1e-6).sum())}/{len(spread)} degenerate"
              f"   <- needs a real spread here, or the pooling risk is untested")
        print(f"  p-values: min={p.min():.4f} median={np.median(p):.3f} "
              f"frac<0.05={np.mean(p < 0.05):.3f} (should be ~0.05)")
        print(f"  KS vs uniform: D={ks.statistic:.3f}, p={ks.pvalue:.4f}")
        print(f"  FALSE discoveries at q<0.05: {int(np.sum(q < 0.05))}/{len(q)}"
              f"   (should be ~0)")
        for m_, pv, qv in zip(common, p, q):
            out.append({"outcome": m_, "kernel_type": kt, "p": pv, "q": qv})
    pd.DataFrame(out).to_csv(f"{args.out_prefix}_pvalues.csv", index=False)
    with open(f"{args.out_prefix}_fit.pkl", "wb") as f:
        pickle.dump(gps, f)
    print(f"\nwrote {args.out_prefix}_{{raw_draws,pvalues}}.csv and _fit.pkl")
    ray.shutdown()


if __name__ == "__main__":
    main()
