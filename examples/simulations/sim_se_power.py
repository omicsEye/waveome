"""SE-component power: can the pipeline recover a KNOWN nonlinear effect?

FINDINGS.md section 20 established that 95% of squared-exponential
components in the reported iHMP fit are numerically dead -- variance exactly
on VARIANCE_FLOOR, lengthscale reverted to the LogNormal(1.0, 0.5) prior
mode (2.117) -- and that their log_bf is therefore a constant (-4.8). The
tested SE strata returned 1/564 and 0/564 significant.

That leaves two indistinguishable explanations, and the manuscript cannot
choose between them from the real data alone:

  (a) correct suppression -- SE terms genuinely do not help once the
      lengthscale prior stops them fitting per-observation noise
  (b) over-suppression   -- the prior kills real nonlinear effects too, and
      the near-zero hit rate reflects no power rather than no signal

This decides it by planting nonlinear effects of known amplitude and
smoothness and asking whether they are recovered. The effect is a draw from
an SE GP with a controlled lengthscale, so the generating process is exactly
the model class the SE kernel is meant to capture -- if the pipeline cannot
recover THAT, it cannot recover anything nonlinear.

Lengthscales are set relative to the covariate's own scale and span the
region around the prior mode, so the output shows WHERE recovery fails, not
merely whether it does:

    ls_rel  0.25  below the data's resolution  (should be suppressed --
                  this is the noise-fitting case the prior was added for)
    ls_rel  0.5   marginal
    ls_rel  1.0   a smooth effect the data can clearly resolve
    ls_rel  2.0   very smooth, near the prior mode

Reported per lengthscale: how often the SE component survives fitting at
all (variance off the floor), its permutation power at q<=0.10, and whether
lin[cindex] picks up the signal instead. True nulls carry no cindex effect
so realized FDR is reported alongside.

    python sim_se_power.py --M 150 --B0 10 --B1 60
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, "1")

import gpflow  # noqa: E402
import ray  # noqa: E402

from waveome.kernels import Lin  # noqa: E402
from waveome.model_search import GPSearch, _component_covariate_names  # noqa: E402
from waveome.utilities import (  # noqa: E402
    VARIANCE_FLOOR,
    calc_bh_qvalues,
    calc_permutation_pvalues,
)

from sim_fdr_stage1_uniformity import between_fraction, draw  # noqa: E402

SEED = 9102
TARGET_BETWEEN_FRAC = 0.41
FRAC_TRUE_POSITIVE = 0.50
LS_REL_GRID = (0.25, 0.5, 1.0, 2.0)


def gp_draw(x, lengthscale, rng):
    """A draw from an SE GP at the observed covariate values, standardized."""
    d = (x[:, None] - x[None, :]) / lengthscale
    K = np.exp(-0.5 * d ** 2) + 1e-8 * np.eye(len(x))
    f = np.linalg.cholesky(K) @ rng.normal(size=len(x))
    return (f - f.mean()) / (f.std() + 1e-12)


def simulate(n_units=49, visit_rate=5, M=150, rng=None):
    rng = rng or np.random.default_rng(SEED)
    n_obs = np.maximum(rng.poisson(visit_rate, n_units), 1)
    unit = np.repeat(np.arange(n_units), n_obs)
    time_v = np.concatenate([np.sort(rng.uniform(-2, 2, k)) for k in n_obs])

    sd_w = 1.0
    sd_b = sd_w * np.sqrt(TARGET_BETWEEN_FRAC / (1 - TARGET_BETWEEN_FRAC))
    cindex = (np.repeat(rng.normal(0, sd_b, n_units), n_obs)
              + rng.normal(0, sd_w, len(unit)))

    X = pd.DataFrame({"id": unit.astype(float), "cindex": cindex,
                      "time": time_v})
    c_sd = cindex.std()
    Y, truth = {}, {}
    for m in range(M):
        sd_u = np.exp(rng.uniform(np.log(0.1), np.log(1.0)))
        u = np.repeat(rng.normal(0, sd_u, n_units), n_obs)
        amp_t, freq = rng.uniform(0.3, 1.0), rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 6.3)
        is_tp = rng.random() < FRAC_TRUE_POSITIVE
        if is_tp:
            ls_rel = LS_REL_GRID[rng.integers(len(LS_REL_GRID))]
            # Amplitudes span weak to strong so there is a power gradient
            # rather than everything being trivially detectable.
            amp_c = np.exp(rng.uniform(np.log(0.2), np.log(1.2)))
            f_c = amp_c * gp_draw(cindex, ls_rel * c_sd, rng)
        else:
            ls_rel, amp_c, f_c = np.nan, 0.0, 0.0
        intercept = rng.uniform(np.log(5.9e3), np.log(1.2e7))
        eta = (intercept + u + amp_t * np.sin(freq * time_v + phase) + f_c)
        Y[f"y{m}"] = rng.poisson(rng.gamma(20.0, np.exp(eta) / 20.0))
        truth[f"y{m}"] = {"is_tp": is_tp, "ls_rel": ls_rel, "amp_c": amp_c}
    return X, pd.DataFrame(Y), pd.DataFrame(truth).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=150)
    ap.add_argument("--B0", type=int, default=10)
    ap.add_argument("--B1", type=int, default=60)
    ap.add_argument("--out-prefix", default="sim_waveome_output/se_power")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    rng = np.random.default_rng(SEED)
    X, Y, truth = simulate(M=args.M, rng=rng)
    truth.to_csv(f"{args.out_prefix}_truth.csv")
    n_tp = int(truth.is_tp.sum())
    print(f"simulated: {X.shape[0]} obs, {args.M} outcomes, {n_tp} true "
          f"positives ({n_tp/args.M:.0%}) with NONLINEAR cindex effects")
    print(f"cindex between-subject fraction: "
          f"{between_fraction(X['cindex'].values, X['id'].values):.3f}")
    print(f"true positives per lengthscale: "
          f"{truth[truth.is_tp].ls_rel.value_counts().to_dict()}\n")

    gps = GPSearch(X=X, Y=Y, unit_col="id", categorical_vars=[],
                   outcome_likelihood="negativebinomial", Y_transform=None)
    t0 = time.time()
    gps.penalized_optimization(
        random_seed=SEED,
        kernel_options={"second_order_numeric": False,
                        "unit_numeric_interactions": False,
                        "categorical_numeric_interactions": False,
                        "kerns": [gpflow.kernels.SquaredExponential(), Lin()]},
        num_restart=3, optimization_options={"optimizer": "scipy"},
        prune_components=False)
    print(f"fitting done in {(time.time()-t0)/60:.1f} min")

    names = list(gps.models.keys())
    kts, cns = _component_covariate_names(gps.models[names[0]].kernel_name,
                                          gps.feat_names)
    targets = [i for i, c in enumerate(cns) if c == "cindex"]
    se_idx = [i for i in targets if kts[i] == "squared_exponential"][0]
    col = gps.feat_names.index("cindex")
    Xn = gps.X.to_numpy()

    # --- did the SE term survive fitting at all? ---
    surv = []
    for n in names:
        k = gps.models[n].kernel.kernels[se_idx]
        surv.append({"outcome": n,
                     "se_variance": float(np.array(k.variance)),
                     "se_lengthscale": float(np.array(k.lengthscales).ravel()[0])})
    surv = pd.DataFrame(surv).set_index("outcome")
    surv["alive"] = surv.se_variance > VARIANCE_FLOOR * 10
    surv = surv.join(truth)
    surv.to_csv(f"{args.out_prefix}_survival.csv")
    print("\n=== does the SE component survive fitting? (variance off floor) ===")
    print(f"  true nulls    : {int(surv[~surv.is_tp.astype(bool)].alive.sum())}"
          f"/{int((~surv.is_tp.astype(bool)).sum())} alive")
    for ls in LS_REL_GRID:
        g = surv[surv.ls_rel == ls]
        if len(g):
            print(f"  ls_rel={ls:<5}: {int(g.alive.sum()):3d}/{len(g):3d} alive"
                  f"   median fitted ls={g[g.alive].se_lengthscale.median():.3g}")

    # --- permutation draws ---
    ray.init(include_dashboard=False, configure_logging=False)
    X_ref = ray.put(Xn)
    refs = {n: (ray.put(gps.models[n]), ray.put(gps.Y[n].to_numpy()))
            for n in names}

    def run(jobs, tag):
        print(f"\n{len(jobs)} {tag} draws queued", flush=True)
        t, rows, done_n, nf = time.time(), [], 0, 0
        pending = [j[2] for j in jobs]
        meta = {j[2]: j[:2] for j in jobs}
        while pending:
            done, pending = ray.wait(pending, num_returns=min(100, len(pending)))
            for ref in done:
                nm, b = meta[ref]
                res, err = ray.get(ref)
                done_n += 1
                if err is not None:
                    nf += 1
                    continue
                for idx, v in res.items():
                    rows.append({"outcome": nm, "kernel_type": kts[idx],
                                 "draw": b, "log_bf": v})
            el = time.time() - t
            print(f"  {done_n}/{len(jobs)} ({el/60:.1f} min, "
                  f"{el/max(done_n,1):.2f} s/draw, {nf} failed)", flush=True)
        return pd.DataFrame(rows)

    obs = run([(n, -1, draw.remote(refs[n][0], X_ref, refs[n][1], col,
                                   targets, None)) for n in names], "observed")
    jobs = [(n, b, draw.remote(refs[n][0], X_ref, refs[n][1], col, targets,
                               SEED + 1000 * b + hash(n) % 1000))
            for n in names for b in range(args.B0)]
    null = run(jobs, f"screen B0={args.B0}")

    # top up only the outcomes whose screen null is not degenerate
    spread = null.groupby(["outcome", "kernel_type"]).log_bf.std()
    live = {(o, k) for (o, k), s in spread.items() if s > 1e-6}
    more = [(n, b, draw.remote(refs[n][0], X_ref, refs[n][1], col, targets,
                               SEED + 1000 * b + hash(n) % 1000))
            for n in names if any((n, k) in live for k in ("squared_exponential", "lin"))
            for b in range(args.B0, args.B1)]
    if more:
        null = pd.concat([null, run(more, f"top-up B1={args.B1}")],
                         ignore_index=True)
    ray.shutdown()
    null.to_csv(f"{args.out_prefix}_draws.csv", index=False)

    # --- p/q values per stratum, then power by lengthscale ---
    print("\n" + "=" * 66)
    for kt in ("squared_exponential", "lin"):
        o = obs[obs.kernel_type == kt].set_index("outcome").log_bf
        nd = {n: g.log_bf.to_numpy()
              for n, g in null[null.kernel_type == kt].groupby("outcome")}
        o = o[[n in nd for n in o.index]]
        pv = calc_permutation_pvalues(o, {n: nd[n] for n in o.index})
        pv["q"] = calc_bh_qvalues(pv["p_value"].to_numpy())
        pv = pv.rename(columns={"p_value": "p"})
        res = truth.join(pv)
        res.to_csv(f"{args.out_prefix}_{kt}_results.csv")
        tp = res.is_tp.astype(bool)
        sel = res.q <= 0.10
        n_sel = int(sel.sum()); n_false = int((sel & ~tp).sum())
        print(f"\n### {kt}[cindex] -- q<=0.10")
        print(f"  discoveries {n_sel}, false {n_false}, "
              f"realized FDR {n_false/n_sel if n_sel else 0:.3f}, "
              f"overall power {int((sel & tp).sum())}/{int(tp.sum())}")
        for ls in LS_REL_GRID:
            g = res[res.ls_rel == ls]
            if len(g):
                print(f"    ls_rel={ls:<5} power {int((g.q <= 0.10).sum()):3d}/{len(g):3d}"
                      f"   median p={g.p.median():.3g}")
    print("\nwrote", args.out_prefix + "_*.csv")


if __name__ == "__main__":
    main()
