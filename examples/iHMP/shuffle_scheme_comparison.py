"""Compare within-subject shuffling against a completely global shuffle.

The claim under test: a global shuffle of hbi across all 238 observations
destroys hbi's subject-level clustering. Real hbi is clustered (41% of its
variance is between-subject), and an SE kernel on a clustered covariate can
partially PROXY the subject random intercept -- gaining likelihood even when
hbi has no relationship to the outcome at all. A global shuffle removes that
ability, so its null log_bf would sit too low, making observed values look
more extreme than they are (anti-conservative -> inflated false discoveries).

If the claim is wrong, the two null distributions will look the same and the
global shuffle is simply the better choice: bigger permutation space, no
design assumptions, far simpler to explain.

Target: HILp_QI2874 -- participant_id log_bf 20.6 (strong subject structure),
SE[hbi] log_bf 36.6. If proxying happens anywhere, it happens here.
"""
import pickle
import time

import gpflow
import numpy as np
import ray
from gpflow.utilities import set_trainable

from waveome.utilities import convert_data_to_tensors

INPUT_FP = (
    "output/fit_penalized_models_revision_full_scipy_ls_prior_no_prune_clamp_removed.pkl"
)
SEED = 9102
B = 15
TARGETS = {5: "SE[hbi]", 6: "lin[hbi]"}
COV_IDX, UNIT_IDX = 2, 0
METABOLITE = "HILp_QI2874"


def shuffle_within(X, rng):
    """Free shuffle of the covariate inside each subject."""
    out = X.copy()
    for u in np.unique(X[:, UNIT_IDX]):
        i = np.where(X[:, UNIT_IDX] == u)[0]
        out[i, COV_IDX] = rng.permutation(X[i, COV_IDX])
    return out


def shuffle_global(X, rng):
    """Free shuffle of the covariate across every observation."""
    out = X.copy()
    out[:, COV_IDX] = rng.permutation(X[:, COV_IDX])
    return out


def between_fraction(x, unit):
    gm = x.mean()
    us = np.unique(unit)
    between = sum(np.sum(unit == u) * (x[unit == u].mean() - gm) ** 2 for u in us)
    within = sum(np.sum((x[unit == u] - x[unit == u].mean()) ** 2) for u in us)
    return between / (between + within)


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
    Xp = {"within": shuffle_within, "global": shuffle_global,
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
    unit = X[:, UNIT_IDX]

    # --- structural check: does the shuffle preserve hbi's clustering? ---
    rng = np.random.default_rng(SEED)
    print(f"between-subject fraction of hbi variance")
    print(f"  real data            : {between_fraction(X[:, COV_IDX], unit):.3f}")
    fw = [between_fraction(shuffle_within(X, rng)[:, COV_IDX], unit) for _ in range(200)]
    fg = [between_fraction(shuffle_global(X, rng)[:, COV_IDX], unit) for _ in range(200)]
    print(f"  after within shuffle : {np.mean(fw):.3f} +/- {np.std(fw):.3f}")
    print(f"  after global shuffle : {np.mean(fg):.3f} +/- {np.std(fg):.3f}")
    print()

    # --- null distributions under each scheme ---
    ray.init(include_dashboard=False, configure_logging=False)
    Xr, mr = ray.put(X), ray.put(gps.models[METABOLITE])
    yr = ray.put(gps.Y[METABOLITE].to_numpy())

    jobs = [("observed", draw.remote(mr, Xr, yr, "observed", 0))]
    for mode in ("within", "global"):
        for b in range(B):
            jobs.append((mode, draw.remote(mr, Xr, yr, mode, SEED + b)))

    print(f"{len(jobs)} fits queued for {METABOLITE}...")
    t0 = time.time()
    res = ray.get([j[1] for j in jobs])
    print(f"done in {(time.time()-t0)/60:.1f} min\n")

    obs = res[0]
    nulls = {"within": {i: [] for i in TARGETS}, "global": {i: [] for i in TARGETS}}
    for (mode, _), r in zip(jobs[1:], res[1:]):
        for i, v in r.items():
            nulls[mode][i].append(v)

    for idx, lab in TARGETS.items():
        print(f"{lab}  observed = {obs[idx]:.2f}")
        for mode in ("within", "global"):
            a = np.array(nulls[mode][idx])
            p = (1 + np.sum(a >= obs[idx])) / (1 + len(a))
            print(f"   {mode:7s} null: mean={a.mean():8.2f}  sd={a.std():6.2f}  "
                  f"min={a.min():8.2f}  max={a.max():8.2f}   p={p:.3f}")
        print()
    ray.shutdown()


if __name__ == "__main__":
    main()
