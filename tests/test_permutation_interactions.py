"""Product components are tested under the permutation of each covariate.

A treat x time component reads both columns, so permuting either one
scrambles it. permutation_significance must therefore test it under BOTH
covariates' permutations, in strata separate from the main effects. A model
with no product components (the iHMP case) must be tested exactly as before:
one component per covariate.

No pytest in this environment, so this runs directly:

    python tests/test_permutation_interactions.py
"""
import os
import sys

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gpflow  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from waveome.model_search import GPSearch  # noqa: E402

SEED = 9102
INTERACTION = "categorical*squared_exponential"


def build_fit(interactions, n_units=12, n_visits=4, n_out=2):
    """Unit-constant treat, within-unit time, optional treat x time term."""
    rng = np.random.default_rng(SEED)
    unit = np.repeat(np.arange(n_units), n_visits)
    treat = np.repeat(np.arange(n_units) % 2, n_visits)
    t = np.tile(np.linspace(-1, 1, n_visits), n_units)
    X = pd.DataFrame({"id": unit.astype(float), "treat": treat.astype(float),
                      "time": t})
    Y = pd.DataFrame({
        f"y{i}": rng.normal(size=len(unit)) + treat * t * (i == 0)
        for i in range(n_out)
    })
    gps = GPSearch(X=X, Y=Y, unit_col="id", categorical_vars=["treat"],
                   outcome_likelihood="gaussian", Y_transform=None)
    gps.penalized_optimization(
        random_seed=SEED,
        kernel_options={"second_order_numeric": False,
                        "unit_numeric_interactions": False,
                        "categorical_numeric_interactions": interactions,
                        "kerns": [gpflow.kernels.SquaredExponential()]},
        optimization_options={"optimizer": "scipy"},
        prune_components=False)
    return gps


def rows_per_draw(draws):
    return draws.groupby(["metabolite", "covariate", "draw"]).size()


def main():
    # --- with a treat x time component ---
    gps = build_fit(interactions=True)
    print(f"kernel: {gps.models['y0'].kernel_name}")
    res = gps.permutation_significance(
        covariates=["treat", "time"], B0=2, B1=2, random_seed=SEED,
        verbose=False)
    strata = set(res["stratum"])
    for cov in ("treat", "time"):
        assert f"{INTERACTION}:{cov}" in strata, \
            f"interaction not tested under the {cov} permutation: {strata}"
    assert "categorical:treat" in strata and "squared_exponential:time" in strata, \
        f"main effects missing: {strata}"
    # Every draw yields the main effect AND the interaction for its covariate
    n = rows_per_draw(gps.permutation_draws)
    assert (n == 2).all(), f"expected 2 components per draw:\n{n}"
    print(f"[ok] interaction tested under both permutations; strata {sorted(strata)}")

    # --- main effects only (the iHMP case): unchanged ---
    gps = build_fit(interactions=False)
    print(f"kernel: {gps.models['y0'].kernel_name}")
    res = gps.permutation_significance(
        covariates=["treat", "time"], B0=2, B1=2, random_seed=SEED,
        verbose=False)
    assert set(res["stratum"]) == {"categorical:treat", "squared_exponential:time"}, \
        f"main-effect strata changed: {set(res['stratum'])}"
    n = rows_per_draw(gps.permutation_draws)
    assert (n == 1).all(), f"expected 1 component per draw:\n{n}"
    print("[ok] main-effect-only model: one component per covariate, as before")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
