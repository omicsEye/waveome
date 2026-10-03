"""Run the empirical-donor-resample null alongside the within-subject shuffle.

Donor resample (the non-parametric form of the (mu,sigma)-swap idea):
permute subject labels, then have each subject draw n_i values WITH
replacement from its donor's own observed hbi values. Breaks both the
between-subject level and the within-subject pairing, needs no matched
block sizes, and never invents a value that isn't a real HBI score.

Run against the within-subject shuffle on the same metabolites so the two
nulls can be compared directly. Expectation from the structural check: the
donor resample over-clusters (between-fraction 0.569 vs a true 0.413), so
its null should sit HIGHER -> larger p-values -> conservative.
"""
import pickle
import time

import gpflow
import numpy as np
import ray
from gpflow.utilities import set_trainable

from waveome.utilities import convert_data_to_tensors

INPUT_FP = (
    "output/ihmp_penalized_fit.pkl"
)
SEED = 9102
B = 20
TARGETS = {5: "SE[hbi]", 6: "lin[hbi]"}
COV_IDX, UNIT_IDX = 2, 0
DEMO = ["HILp_QI2874", "HILn_QI112", "C18n_QI41", "C8p_QI140"]


def shuffle_within(X, rng):
    out = X.copy()
    for u in np.unique(X[:, UNIT_IDX]):
        i = np.where(X[:, UNIT_IDX] == u)[0]
        out[i, COV_IDX] = rng.permutation(X[i, COV_IDX])
    return out


def donor_resample(X, rng):
    out = X.copy()
    us = np.unique(X[:, UNIT_IDX])
    donors = rng.permutation(us)
    for u, d in zip(us, donors):
        i = np.where(X[:, UNIT_IDX] == u)[0]
        pool = X[X[:, UNIT_IDX] == d, COV_IDX]
        out[i, COV_IDX] = rng.choice(pool, size=len(i), replace=True)
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
def draw(model, X, y, mode, seed):
    rng = np.random.default_rng(seed)
    Xp = {"within": shuffle_within, "donor": donor_resample,
          "observed": lambda x, r: x}[mode](X, rng)
    data = convert_data_to_tensors(Xp, y.reshape(-1, 1))
    opt = getattr(model, "optimizer", None) or "scipy"
    full = gpflow.utilities.deepcopy(model)
    full.num_trainable_params = np.nan
    full.optimize_params(data=data, optimizer=opt)
    bf = _bic(full, data)
    out = {}
    for idx in TARGETS:
        red = gpflow.utilities.deepcopy(full)
        red.kernel.kernels.pop(idx)
        red.num_trainable_params = np.nan
        red.optimize_params(data=data, optimizer=opt)
        out[idx] = float(-0.5 * (bf - _bic(red, data)))
    return out


def main():
    with open(INPUT_FP, "rb") as f:
        gps = pickle.load(f)
    X = gps.X.to_numpy()
    ray.init(include_dashboard=False, configure_logging=False)
    Xr = ray.put(X)

    jobs = []
    for name in DEMO:
        mr = ray.put(gps.models[name])
        yr = ray.put(gps.Y[name].to_numpy())
        jobs.append((name, "observed", draw.remote(mr, Xr, yr, "observed", 0)))
        for mode in ("within", "donor"):
            for b in range(B):
                jobs.append((name, mode,
                             draw.remote(mr, Xr, yr, mode,
                                         SEED + 1000 * DEMO.index(name) + b)))

    print(f"{len(jobs)} fits queued...")
    t0 = time.time()
    res = ray.get([j[2] for j in jobs])
    print(f"done in {(time.time()-t0)/60:.1f} min\n")

    obs, nulls = {}, {}
    for (name, mode, _), r in zip(jobs, res):
        if mode == "observed":
            obs[name] = r
        else:
            nulls.setdefault((name, mode), {i: [] for i in TARGETS})
            for i, v in r.items():
                nulls[(name, mode)][i].append(v)

    for idx, lab in TARGETS.items():
        print(f"===== {lab} =====")
        print(f"{'metabolite':13s}{'observed':>10s}  {'scheme':7s}"
              f"{'null mean':>11s}{'null sd':>9s}{'null max':>10s}{'p':>8s}")
        for name in DEMO:
            o = obs[name][idx]
            for mode in ("within", "donor"):
                a = np.array(nulls[(name, mode)][idx])
                p = (1 + np.sum(a >= o)) / (1 + len(a))
                lead = f"{name:13s}{o:10.2f}" if mode == "within" else " " * 23
                print(f"{lead}  {mode:7s}{a.mean():11.2f}{a.std():9.2f}"
                      f"{a.max():10.2f}{p:8.3f}")
        print()
    ray.shutdown()


if __name__ == "__main__":
    main()
