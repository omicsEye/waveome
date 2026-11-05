"""The between-subject refit drops every unit-specific component.

Its residuals must keep each unit's mean, which is what the test correlates
with the unit-level covariate. A unit kernel left in the refit absorbs part
of that mean, shrinking it by sigma2_e / (sigma2_e + n_u sigma2_unit): with
unequal visit counts n_u the shrinkage differs by unit, reorders the units
and costs power. Validity does not depend on the choice -- the covariate is
absent from the refit and the units are permuted -- so the null check here
is only a guard against gross error.

No pytest in this environment, so this runs directly:

    python tests/test_between_subject_unit_components.py
"""
import os
import sys

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gpflow  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from waveome.model_search import GPSearch, _between_subject_drop_idx  # noqa: E402

SEED = 9102
N_SIGNAL, N_NULL = 6, 6


def check_drop_idx():
    names = ["id", "treat", "z", "id*time", "z*time", "time"]
    new = _between_subject_drop_idx(names, "z", "id", True)
    old = _between_subject_drop_idx(names, "z", "id", False)
    assert new == [0, 2, 3, 4], new
    assert old == [2], old
    print("[ok] drops the unit term, the covariate, and their interactions")


def build_fit(n_units=40):
    """Unit-level covariate z; a strong unit intercept; 1-8 visits per unit.

    time varies within units and has no effect; it stays in the refit, as
    the iHMP model's other terms do.
    """
    rng = np.random.default_rng(SEED)
    visits = rng.integers(1, 9, size=n_units)
    unit = np.repeat(np.arange(n_units), visits)
    z_u = rng.normal(size=n_units)
    b_u = rng.normal(size=n_units)
    X = pd.DataFrame({"id": unit.astype(float), "z": z_u[unit],
                      "time": rng.uniform(-1, 1, size=len(unit))})
    Y = {}
    for i in range(N_SIGNAL + N_NULL):
        b = rng.normal(size=n_units)          # each outcome its own unit effect
        effect = 0.8 if i < N_SIGNAL else 0.0
        Y[f"y{i}"] = effect * z_u[unit] + b[unit] + rng.normal(scale=0.5, size=len(unit))
    gps = GPSearch(X=X, Y=pd.DataFrame(Y), unit_col="id",
                   outcome_likelihood="gaussian", Y_transform=None)
    gps.penalized_optimization(
        random_seed=SEED,
        kernel_options={"second_order_numeric": False,
                        "unit_numeric_interactions": False,
                        "categorical_numeric_interactions": False,
                        "kerns": [gpflow.kernels.SquaredExponential()]},
        optimization_options={"optimizer": "scipy"},
        prune_components=False)
    return gps, visits


def main():
    check_drop_idx()

    gps, visits = build_fit()
    print(f"kernel: {gps.models['y0'].kernel_name}; visits {visits.min()}-{visits.max()}")
    res = {}
    for flag in (False, True):
        r = gps.between_subject_significance(
            covariates=["z"], n_permutations=2000, random_seed=SEED,
            drop_unit_components=flag, verbose=False)
        r["signal"] = r.metabolite.str[1:].astype(int) < N_SIGNAL
        res[flag] = r.set_index("metabolite")
    old, new = res[False], res[True]
    print(pd.DataFrame({"rho_old": old.spearman_abs, "p_old": old.p_value,
                        "rho_new": new.spearman_abs, "p_new": new.p_value,
                        "signal": new.signal}).round(4).to_string())

    sig = new.signal
    assert new.spearman_abs[sig].mean() > old.spearman_abs[sig].mean(), \
        "dropping the unit term did not strengthen the between-unit signal"
    assert (new.p_value[sig] < 0.05).sum() >= (old.p_value[sig] < 0.05).sum(), \
        "dropping the unit term detected fewer true effects"
    print(f"[ok] signal: mean |rho| {old.spearman_abs[sig].mean():.3f} -> "
          f"{new.spearman_abs[sig].mean():.3f}; p<0.05 "
          f"{(old.p_value[sig] < 0.05).sum()} -> {(new.p_value[sig] < 0.05).sum()} "
          f"of {N_SIGNAL}")

    n_false = int((new.p_value[~sig] < 0.05).sum())
    assert n_false <= 2, f"{n_false} of {N_NULL} null outcomes at p<0.05"
    print(f"[ok] null: {n_false} of {N_NULL} at p<0.05")

    # A refit with nothing left (unit term and covariate were the whole
    # kernel) returns the centred outcome rather than failing
    import ray
    from waveome.model_search import _subject_residuals_remote
    ray.init(num_cpus=1, ignore_reinit_error=True)
    y = gps.Y["y0"].to_numpy()
    n_comp = len(gps.models["y0"].kernel.kernels)
    r, err = ray.get(_subject_residuals_remote.remote(
        gps.models["y0"], gps.X.to_numpy(), y, list(range(n_comp)), "pearson"))
    ray.shutdown()
    assert err is None and np.allclose(r, y - y.mean()), err
    print("[ok] refit with every component dropped returns the centred outcome")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
