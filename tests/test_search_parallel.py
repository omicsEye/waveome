"""Kernel search fitted in parallel selects exactly what the sequential one does.

loc_kernel_search's candidates are fitted as concurrent Ray tasks when Ray is
running, one after another otherwise. Each fit is seeded from the search seed
and the candidate's kernel name, so the two must agree candidate for
candidate. The data have more observations than inducing points (500), so
every fit draws a random inducing subset: without the per-candidate seed the
parallel fits would not reproduce.

No pytest in this environment, so this runs directly:

    python tests/test_search_parallel.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, "1")

import gpflow  # noqa: E402
import ray  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from waveome.model_search import full_kernel_search  # noqa: E402

SEED = 9102


def make_data(n_units=52, n_visits=10):
    rng = np.random.default_rng(SEED)
    unit = np.repeat(np.arange(n_units), n_visits)
    t = np.tile(np.linspace(-1.5, 1.5, n_visits), n_units)
    y = (np.repeat(rng.normal(0, 0.7, n_units), n_visits) + np.sin(2 * t)
         + rng.normal(0, 0.5, len(unit)))
    return (pd.DataFrame({"id": unit.astype(float), "time": t}),
            pd.DataFrame({"y": y}))


def search(X, Y):
    out = full_kernel_search(
        X=X, Y=Y, kern_list=[gpflow.kernels.SquaredExponential()],
        cat_vars=[0], max_depth=2, num_restart=1, lik="gaussian",
        random_seed=SEED, keep_only_best=False)
    return out["best_model"], {k: v["bic"] for k, v in out["models"].items()}


def main():
    X, Y = make_data()
    assert len(X) > 500, "need more observations than inducing points"

    t0 = time.time()
    best_seq, bic_seq = search(X, Y)
    t_seq = time.time() - t0
    print(f"sequential: best {best_seq}, {len(bic_seq)} models, {t_seq:.0f}s")

    ray.init(num_cpus=4, include_dashboard=False)
    try:
        t0 = time.time()
        best_par, bic_par = search(X, Y)
        t_par = time.time() - t0
    finally:
        ray.shutdown()
    print(f"parallel:   best {best_par}, {len(bic_par)} models, {t_par:.0f}s")

    assert best_par == best_seq, f"best model differs: {best_par} vs {best_seq}"
    assert bic_par.keys() == bic_seq.keys(), \
        f"candidate sets differ: {sorted(bic_par)} vs {sorted(bic_seq)}"
    diff = {k: (bic_seq[k], bic_par[k]) for k in bic_seq if bic_seq[k] != bic_par[k]}
    assert not diff, f"BIC differs for {diff}"
    print(f"\n[ok] identical best model and BIC for all {len(bic_seq)} models")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
