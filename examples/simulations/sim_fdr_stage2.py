"""T2 Stage 2 -- realized FDR vs nominal, with known true positives.

This is the check the reviewers actually asked for (R1.M5 / R2.5: "no
calibration / FDR control ... report realized FDR/FWER vs nominal +
sensitivity over q"), and it doubles as the first real test of the GPD tail
approximation, which Stage 1 could not exercise: GPD only triggers when an
observed statistic sits in the extreme tail, and a null-only simulation has
essentially none.

Outcomes are labelled:
  * TRUE NULL     -- no within-subject cindex effect. May still carry a
                     between-subject one; H0-within is true either way, and
                     that between effect is what produces the wide,
                     heterogeneous permutation nulls.
  * TRUE POSITIVE -- a real within-subject cindex effect.

Two p-value constructions are scored side by side on the same draws:
  (a) SCALE-STRATIFIED pooling  -- validated in Stage 1 (wide-subgroup
      false-positive rate 0.032 / 0.028 against nominal 0.05)
  (b) GPD TAIL per outcome      -- no pooling at all; each test uses only
      its own permutations (Knijnenburg et al. 2009)

Stage 1's GPD probe found 73% of fits returning shape c < 0, i.e. a BOUNDED
tail, and one p-value of exactly 0 -- invalid, and maximally significant
under BH. Real discoveries sit far out in the tail, which is exactly where
a bounded fit fails, so the guard here falls back to the empirical p-value
whenever the fitted tail probability collapses to zero. How often that
fires is reported.

    python sim_fdr_stage2.py --M 400 --B0 10 --B1 100
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
from scipy.stats import genpareto, goodness_of_fit  # noqa: E402

from waveome.kernels import Lin  # noqa: E402
from waveome.model_search import GPSearch, _component_covariate_names  # noqa: E402
from waveome.utilities import calc_bh_qvalues  # noqa: E402

from sim_fdr_stage1_uniformity import between_fraction, draw  # noqa: E402

SEED = 9102
UNIT_IDX = 0
TIE_TOL = 1e-3
DEGEN_TOL = 1e-6
WIDE_TOL = 0.5
EMPIRICAL_MIN_EXC = 10
AD_ALPHA = 0.05
TARGET_BETWEEN_FRAC = 0.41
FRAC_TRUE_POSITIVE = 0.30


def simulate(n_units=49, visit_rate=5, M=400, rng=None):
    rng = rng or np.random.default_rng(SEED)
    n_obs = np.maximum(rng.poisson(visit_rate, n_units), 1)
    unit = np.repeat(np.arange(n_units), n_obs)
    time_v = np.concatenate([np.sort(rng.uniform(-2, 2, k)) for k in n_obs])

    sd_w = 1.0
    sd_b = sd_w * np.sqrt(TARGET_BETWEEN_FRAC / (1 - TARGET_BETWEEN_FRAC))
    cindex = (np.repeat(rng.normal(0, sd_b, n_units), n_obs)
              + rng.normal(0, sd_w, len(unit)))
    c_between = np.repeat(
        [cindex[unit == u].mean() for u in np.arange(n_units)], n_obs)
    c_within = cindex - c_between

    X = pd.DataFrame({"id": unit.astype(float), "cindex": cindex,
                      "time": time_v})
    Y, truth = {}, {}
    for m in range(M):
        sd_u = np.exp(rng.uniform(np.log(0.1), np.log(1.0)))
        u = np.repeat(rng.normal(0, sd_u, n_units), n_obs)
        amp, freq = rng.uniform(0.3, 1.0), rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 6.3)
        beta_b = 0.0 if rng.random() < 0.3 else np.exp(
            rng.uniform(np.log(0.2), np.log(1.5)))
        is_tp = rng.random() < FRAC_TRUE_POSITIVE
        # Effect sizes span weak to strong so the FDR table has a real
        # power gradient rather than everything being trivially detectable.
        beta_w = np.exp(rng.uniform(np.log(0.15), np.log(1.2))) if is_tp else 0.0
        intercept = rng.uniform(np.log(5.9e3), np.log(1.2e7))
        eta = (intercept + u + amp * np.sin(freq * time_v + phase)
               + beta_b * c_between + beta_w * c_within)
        mu = np.exp(eta)
        Y[f"y{m}"] = rng.poisson(rng.gamma(20.0, mu / 20.0))
        truth[f"y{m}"] = {"is_tp": is_tp, "beta_within": beta_w,
                          "beta_between": beta_b}
    return X, pd.DataFrame(Y), pd.DataFrame(truth).T


def pvals_stratified(obs, centre, spread, by_out):
    def binof(s):
        return 0 if s < DEGEN_TOL else (1 if s < WIDE_TOL else 2)
    bins = {o: binof(spread[o]) for o in obs.index}
    pooled = {0: [], 1: [], 2: []}
    for o, v in by_out.items():
        pooled[bins[o]].extend(np.asarray(v) - centre[o])
    pooled = {k: np.asarray(v) for k, v in pooled.items()}
    return pd.Series({o: (1 + np.sum(pooled[bins[o]] >= obs[o] - centre[o]
                                     - TIE_TOL)) / (1 + len(pooled[bins[o]]))
                      for o in obs.index})


def gpd_pvalue(null, obs, n_exc_start=40, n_exc_min=8, step=4):
    null = np.asarray(null, float)
    B = len(null)
    n_over = int(np.sum(null >= obs - TIE_TOL))
    emp = (1 + n_over) / (1 + B)
    if n_over >= EMPIRICAL_MIN_EXC:
        return emp, "empirical"
    srt = np.sort(null)
    for n_exc in range(min(n_exc_start, B // 2), n_exc_min - 1, -step):
        u = srt[-n_exc - 1]
        exc = srt[-n_exc:] - u
        if np.ptp(exc) < TIE_TOL:
            continue
        try:
            res = goodness_of_fit(genpareto, exc, known_params={"loc": 0},
                                  statistic="ad", n_mc_samples=199,
                                  random_state=np.random.default_rng(SEED))
        except Exception:
            continue
        if res.pvalue > AD_ALPHA:
            c, _, scale = genpareto.fit(exc, floc=0)
            tail = genpareto.sf(obs - u, c, loc=0, scale=scale)
            # c<0 gives a BOUNDED tail; anything past the endpoint returns
            # sf=0, which is not a small p-value but an invalid one.
            if not np.isfinite(tail) or tail <= 0:
                return emp, "gpd-bounded-fallback"
            return (n_exc / B) * tail, "gpd"
    return emp, "empirical-fallback"


def fdr_table(p, truth_tp, label):
    print(f"\n  --- {label} ---")
    print(f"  {'q':>6s}{'discoveries':>13s}{'false':>7s}"
          f"{'realized FDR':>14s}{'power':>8s}")
    n_tp = int(truth_tp.sum())
    for q in (0.01, 0.05, 0.10):
        qv = calc_bh_qvalues(p.values)
        sel = qv < q
        n_sel = int(sel.sum())
        n_false = int((sel & ~truth_tp.values).sum())
        fdp = n_false / n_sel if n_sel else 0.0
        pwr = int((sel & truth_tp.values).sum()) / n_tp if n_tp else 0.0
        flag = "  <- ABOVE nominal" if fdp > q else ""
        print(f"  {q:>6.2f}{n_sel:>13d}{n_false:>7d}{fdp:>14.3f}{pwr:>8.3f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=400)
    ap.add_argument("--B0", type=int, default=10)
    ap.add_argument("--B1", type=int, default=100)
    ap.add_argument("--out-prefix", default="sim_waveome_output/t2_stage2")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    X, Y, truth = simulate(M=args.M, rng=rng)
    truth.to_csv(f"{args.out_prefix}_truth.csv")
    n_tp = int(truth.is_tp.sum())
    print(f"simulated: {X.shape[0]} obs, {args.M} outcomes, "
          f"{n_tp} true positives ({n_tp/args.M:.0%}), {args.M-n_tp} true nulls")
    print(f"cindex between-subject fraction: "
          f"{between_fraction(X['cindex'].values, X['id'].values):.3f}")

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
    col = gps.feat_names.index("cindex")
    Xn = gps.X.to_numpy()

    ray.init(include_dashboard=False, configure_logging=False)
    X_ref = ray.put(Xn)
    refs = {n: (ray.put(gps.models[n]), ray.put(gps.Y[n].to_numpy()))
            for n in names}

    def run(jobs, tag):
        print(f"\n{len(jobs)} {tag} draws queued", flush=True)
        t = time.time()
        rows, done_n, nf = [], 0, 0
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
                  f"{el/done_n:.2f} s/draw, {nf} failed)", flush=True)
        return pd.DataFrame(rows)

    jobs = []
    for mi, n in enumerate(names):
        m_ref, y_ref = refs[n]
        jobs.append((n, -1, draw.remote(m_ref, X_ref, y_ref, col, targets, None)))
        for b in range(args.B0):
            jobs.append((n, b, draw.remote(m_ref, X_ref, y_ref, col, targets,
                                           SEED + 100003 * mi + b)))
    d = run(jobs, "screen")

    sd = d[d.draw >= 0].groupby(["outcome", "kernel_type"])["log_bf"].std().unstack()
    need = sd.index[(sd >= DEGEN_TOL).any(axis=1)].tolist()
    print(f"\ntopping up {len(need)}/{len(names)} non-degenerate outcomes "
          f"to B={args.B1}")
    jobs = []
    for n in need:
        mi = names.index(n)
        m_ref, y_ref = refs[n]
        for b in range(args.B0, args.B1):
            jobs.append((n, b, draw.remote(m_ref, X_ref, y_ref, col, targets,
                                           SEED + 100003 * mi + b)))
    d = pd.concat([d, run(jobs, "top-up")], ignore_index=True)
    d.to_csv(f"{args.out_prefix}_raw_draws.csv", index=False)

    for kt, g in d.groupby("kernel_type"):
        obs = g[g.draw == -1].set_index("outcome")["log_bf"]
        nul = g[g.draw >= 0]
        centre = nul.groupby("outcome")["log_bf"].median()
        spread = nul.groupby("outcome")["log_bf"].std()
        by_out = {o: v["log_bf"].values for o, v in nul.groupby("outcome")}
        common = [o for o in obs.index if o in by_out]
        obs, centre, spread = obs[common], centre[common], spread[common]
        by_out = {o: by_out[o] for o in common}
        tp = truth.loc[common, "is_tp"].astype(bool)

        print(f"\n================ {kt}:cindex ================")
        p_a = pvals_stratified(obs, centre, spread, by_out)
        fdr_table(p_a, tp, "(a) scale-stratified pooling")

        res = {o: gpd_pvalue(by_out[o], obs[o]) if spread[o] >= DEGEN_TOL
               else ((1.0 if obs[o] - centre[o] <= TIE_TOL
                      else 1.0 / (1 + len(by_out[o]))), "degenerate")
               for o in common}
        p_b = pd.Series({o: v[0] for o, v in res.items()})
        meth = pd.Series({o: v[1] for o, v in res.items()})
        print("\n  GPD method usage: " + ", ".join(
            f"{k}={v}" for k, v in meth.value_counts().items()))
        fdr_table(p_b, tp, "(b) GPD tail, no pooling")

        out = pd.DataFrame({"p_stratified": p_a, "p_gpd": p_b,
                            "method": meth, "null_sd": spread,
                            "is_tp": tp})
        out.to_csv(f"{args.out_prefix}_{kt}_pvalues.csv")

    with open(f"{args.out_prefix}_fit.pkl", "wb") as f:
        pickle.dump(gps, f)
    print(f"\nwrote {args.out_prefix}_* ")
    ray.shutdown()


if __name__ == "__main__":
    main()
