"""A live observed statistic against a collapsed null must be topped up.

The adaptive allocation skips the top-up for a component whose screening
null is a point mass, on the grounds that more draws cannot change its
p-value. That holds only when the observed value sits ON the point mass. When
it sits above, the p-value is 1/(1 + n draws), so stopping at B0=10 caps it at
0.09 -- a level BH essentially never rejects at.

The screen is fabricated as a checkpoint (resume=True skips recomputing it):
one component has observed +5 against a null fixed at -2, every other
component's observed sits on its null. Only the first may be topped up.

No pytest in this environment, so this runs directly:

    python tests/test_permutation_collapsed_null.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gpflow  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from waveome.model_search import GPSearch  # noqa: E402

SEED = 9102
B0, B1 = 3, 6
COMPONENTS = {"treat": ("categorical", 1), "time": ("squared_exponential", 2)}
LIVE = ("y0", "time")


def build_fit(n_units=12, n_visits=4, n_out=2):
    rng = np.random.default_rng(SEED)
    unit = np.repeat(np.arange(n_units), n_visits)
    X = pd.DataFrame({"id": unit.astype(float),
                      "treat": np.repeat(np.arange(n_units) % 2, n_visits).astype(float),
                      "time": np.tile(np.linspace(-1, 1, n_visits), n_units)})
    Y = pd.DataFrame({f"y{i}": rng.normal(size=len(unit)) for i in range(n_out)})
    gps = GPSearch(X=X, Y=Y, unit_col="id", categorical_vars=["treat"],
                   outcome_likelihood="gaussian", Y_transform=None)
    gps.penalized_optimization(
        random_seed=SEED,
        kernel_options={"second_order_numeric": False,
                        "unit_numeric_interactions": False,
                        "categorical_numeric_interactions": False,
                        "kerns": [gpflow.kernels.SquaredExponential()]},
        optimization_options={"optimizer": "scipy"},
        prune_components=False)
    return gps


def main():
    tmp = tempfile.mkdtemp()
    ck = os.path.join(tmp, "draws.csv")
    try:
        gps = build_fit()
        rows = [
            {"metabolite": m, "covariate": c, "kernel_type": kt, "kernel_idx": ki,
             "draw": b,
             "log_bf": 5.0 if (m, c) == LIVE and b == -1 else -2.0}
            for m in gps.models for c, (kt, ki) in COMPONENTS.items()
            for b in range(-1, B0)
        ]
        pd.DataFrame(rows).to_csv(ck, index=False)

        gps.permutation_significance(
            covariates=list(COMPONENTS), B0=B0, B1=B1, random_seed=SEED,
            checkpoint_path=ck, resume=True, verbose=False)
        n = (gps.permutation_draws[gps.permutation_draws.draw >= 0]
             .groupby(["metabolite", "covariate"]).size())
        print(n.to_string())
        assert n[LIVE] == B1, \
            f"live observed vs collapsed null got {n[LIVE]} draws, want {B1}"
        others = n.drop(LIVE)
        assert (others == B0).all(), \
            f"observed on its collapsed null should stay at B0={B0}:\n{others}"
        print(f"\n[ok] topped up only the live-vs-collapsed component to B1={B1}")
    finally:
        shutil.rmtree(tmp)

    print("\nall checks passed")


if __name__ == "__main__":
    main()
