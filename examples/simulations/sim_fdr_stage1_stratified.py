"""T2 Stage 1, second attempt: fix the scale-heterogeneity failure.

Pooling one shared tail across all outcomes gave a 37-38% false-positive
rate at nominal 5% for outcomes whose own permutation null is wide, while
the aggregate read 0.047/0.033 because the degenerate majority diluted it.
An outcome with null SD ~19 was being compared against a tail built mostly
from outcomes whose null is a point mass.

Combines the two candidate fixes and scores them side by side:

  (a) SCALE-STRATIFIED pooling -- bin outcomes by their own null SD
      (degenerate / narrow / wide) and pool centred draws only within bin,
      so an outcome is never compared against a tail from a different
      scale regime.

  (b) STANDARDISED pooling -- divide each outcome's excess by its own null
      SD and pool the standardised draws, which assumes a common null
      SHAPE rather than a common scale. Degenerate outcomes have SD 0 and
      are handled by their own point-mass null instead.

ADAPTIVE B: the degenerate outcomes are self-resolving (excess 0 -> p ~ 1)
and need no tail resolution, so extra draws are spent only on outcomes
whose null is non-degenerate. Reuses the B=20 draws and fitted models
already produced by sim_fdr_stage1_uniformity.py rather than refitting.

    python sim_fdr_stage1_stratified.py --B1 60

!!! These are PROTOTYPE implementations kept for method comparison. They
are NOT the shipped construction -- `waveome.utilities.calc_permutation_pvalues`
is. They differ in real ways (the prototype quantile regression here lacks the
shipped monotonisation of the fitted quantile curve, among others), so numbers
from this script must NOT be cited as calibration of the reported method. See
FINDINGS.md section 24. `tests/test_sims_use_shipped_pvalues.py` enforces the
split between comparison scripts and calibration ones.
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

from waveome.model_search import _component_covariate_names  # noqa: E402
from waveome.utilities import calc_bh_qvalues  # noqa: E402

from sim_fdr_stage1_uniformity import draw  # noqa: E402

SEED = 9102
PREFIX = "sim_waveome_output/t2_stage1v2"      # default; override with --prefix
OUT = "sim_waveome_output/t2_stage1strat"     # default; override with --out
TIE_TOL = 1e-3
DEGEN_TOL = 1e-6      # null SD below this = point mass
WIDE_TOL = 0.5        # bin boundary between narrow and wide


def pvals_stratified(obs, centre, spread, draws_by_out):
    """(a) pool centred draws only within a scale bin."""
    def binof(s):
        return 0 if s < DEGEN_TOL else (1 if s < WIDE_TOL else 2)
    bins = {o: binof(spread[o]) for o in obs.index}
    pooled = {0: [], 1: [], 2: []}
    for o, vals in draws_by_out.items():
        pooled[bins[o]].extend(np.asarray(vals) - centre[o])
    pooled = {k: np.asarray(v) for k, v in pooled.items()}
    p = {}
    for o in obs.index:
        pool = pooled[bins[o]]
        e = obs[o] - centre[o]
        p[o] = (1 + np.sum(pool >= e - TIE_TOL)) / (1 + len(pool))
    return pd.Series(p), pd.Series(bins)


def pvals_standardised(obs, centre, spread, draws_by_out):
    """(b) pool z-scores; degenerate outcomes use their own point mass."""
    pool = []
    for o, vals in draws_by_out.items():
        if spread[o] >= DEGEN_TOL:
            pool.extend((np.asarray(vals) - centre[o]) / spread[o])
    pool = np.asarray(pool)
    p = {}
    for o in obs.index:
        e = obs[o] - centre[o]
        if spread[o] < DEGEN_TOL:
            # point-mass null: either the observed sits on it, or the
            # permutation floor is the strongest statement available
            n_own = len(draws_by_out[o])
            p[o] = 1.0 if e <= TIE_TOL else 1.0 / (1 + n_own)
        else:
            z = e / spread[o]
            p[o] = (1 + np.sum(pool >= z)) / (1 + len(pool))
    return pd.Series(p)


def report(name, p, spread, kt):
    print(f"\n  [{name}]  frac(p<0.05) overall = {np.mean(p < 0.05):.3f}")
    q = calc_bh_qvalues(p.values)
    print(f"      false discoveries at q<0.05: {int(np.sum(q < 0.05))}/{len(q)}")
    for lab, m in [("degenerate", spread < DEGEN_TOL),
                   ("narrow", (spread >= DEGEN_TOL) & (spread < WIDE_TOL)),
                   ("WIDE", spread >= WIDE_TOL)]:
        s = p[m.reindex(p.index).fillna(False)]
        if len(s) == 0:
            print(f"      {lab:11s} n=  0")
            continue
        print(f"      {lab:11s} n={len(s):3d}  frac(p<0.05)={np.mean(s < 0.05):.3f}"
              f"   <- nominal 0.05")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B1", type=int, default=60,
                    help="total draws for non-degenerate outcomes")
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    prefix, out = args.prefix, args.out

    with open(f"{prefix}_fit.pkl", "rb") as f:
        gps = pickle.load(f)
    old = pd.read_csv(f"{prefix}_raw_draws.csv")
    B0 = old[old.draw >= 0].draw.max() + 1
    print(f"reusing {len(old)} existing draw-rows (B0={B0}) and {len(gps.models)} fits")

    names = list(gps.models.keys())
    kts, cns = _component_covariate_names(gps.models[names[0]].kernel_name,
                                          gps.feat_names)
    targets = [i for i, c in enumerate(cns) if c == "cindex"]
    col = gps.feat_names.index("cindex")

    # Which outcomes need more draws? Any that is non-degenerate for EITHER
    # component -- a single draw computes both, so the union is what matters.
    sd = (old[old.draw >= 0].groupby(["outcome", "kernel_type"])["log_bf"]
          .std().unstack())
    need = sd.index[(sd >= DEGEN_TOL).any(axis=1)].tolist()
    print(f"topping up {len(need)}/{len(names)} non-degenerate outcomes "
          f"from B={B0} to B={args.B1}")
    print(f"({len(names)-len(need)} degenerate outcomes need no extra draws -- "
          f"their excess is 0 so p ~ 1 regardless)")

    ray.init(include_dashboard=False, configure_logging=False)
    X_ref = ray.put(gps.X.to_numpy())
    jobs = []
    for name in need:
        mi = names.index(name)
        m_ref = ray.put(gps.models[name])
        y_ref = ray.put(gps.Y[name].to_numpy())
        for b in range(B0, args.B1):
            jobs.append((name, b, draw.remote(m_ref, X_ref, y_ref, col,
                                              targets, SEED + 100003 * mi + b)))

    print(f"\n{len(jobs)} extra draws queued")
    t0 = time.time()
    rows, done_n, n_fail = [], 0, 0
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

    alld = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    alld.to_csv(f"{out}_raw_draws.csv", index=False)

    for kt, g in alld.groupby("kernel_type"):
        obs = g[g.draw == -1].set_index("outcome")["log_bf"]
        nul = g[g.draw >= 0]
        centre = nul.groupby("outcome")["log_bf"].median()
        spread = nul.groupby("outcome")["log_bf"].std()
        by_out = {o: v["log_bf"].values for o, v in nul.groupby("outcome")}
        common = obs.index.intersection(centre.index)
        obs, centre, spread = obs[common], centre[common], spread[common]
        by_out = {o: by_out[o] for o in common}

        print(f"\n=== {kt}:cindex ===")
        print(f"  draws per outcome: min={min(len(v) for v in by_out.values())} "
              f"max={max(len(v) for v in by_out.values())}")
        p_a, _ = pvals_stratified(obs, centre, spread, by_out)
        report("(a) scale-stratified", p_a, spread, kt)
        p_b = pvals_standardised(obs, centre, spread, by_out)
        report("(b) standardised", p_b, spread, kt)

    ray.shutdown()


if __name__ == "__main__":
    main()
