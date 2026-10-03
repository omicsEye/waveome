"""Regression test: resuming a checkpoint whose scope differs from the call.

Three bugs shipped in the checkpoint/resume support because its only test
resumed the SAME call that wrote the checkpoint. That path works; the paths
that fail are the ones checkpointing exists to enable -- topping up one
covariate from a multi-covariate run, or re-running a subset of outcomes:

  1. the completeness check indexed `targets` by a covariate the current call
     does not target -> KeyError, crashing on launch
  2. the top-up set was computed over outcomes absent from `names` ->
     names.index() would raise for any that needed draws
  3. the truncated-row counter also counted out-of-scope rows, reporting
     "40308 truncated row(s) dropped" for rows that were merely filtered

No pytest in this environment, so this runs directly:

    python tests/test_permutation_resume.py
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
from waveome.kernels import Lin  # noqa: E402
from waveome.model_search import GPSearch  # noqa: E402

SEED = 9102


def build_fit(n_units=12, n_visits=3, n_out=3):
    """Smallest fit that still has two covariates and repeated measures."""
    rng = np.random.default_rng(SEED)
    unit = np.repeat(np.arange(n_units), n_visits)
    t = np.tile(np.linspace(-1, 1, n_visits), n_units)
    a = np.repeat(rng.normal(size=n_units), n_visits) + rng.normal(0, 0.5, len(unit))
    b = rng.normal(size=len(unit))
    X = pd.DataFrame({"id": unit.astype(float), "cov_a": a, "cov_b": b, "t": t})
    Y = pd.DataFrame({
        f"y{i}": rng.normal(size=len(unit)) + 0.5 * a * (i == 0)
        for i in range(n_out)
    })
    gps = GPSearch(X=X, Y=Y, unit_col="id", categorical_vars=[],
                   outcome_likelihood="gaussian", Y_transform=None)
    gps.penalized_optimization(
        random_seed=SEED,
        kernel_options={"second_order_numeric": False,
                        "unit_numeric_interactions": False,
                        "categorical_numeric_interactions": False,
                        "kerns": [gpflow.kernels.SquaredExponential(), Lin()]},
        num_restart=1, optimization_options={"optimizer": "scipy"},
        prune_components=False)
    return gps


def main():
    tmp = tempfile.mkdtemp()
    ck = os.path.join(tmp, "draws.csv")
    try:
        gps = build_fit()
        names = list(gps.models)
        print(f"fitted {len(names)} outcomes\n")

        # --- write a checkpoint covering BOTH covariates ---
        full = gps.permutation_significance(
            covariates=["cov_a", "cov_b"], B0=2, B1=2, random_seed=SEED,
            checkpoint_path=ck, verbose=False)
        on_disk = pd.read_csv(ck)
        n_all = len(on_disk)
        assert set(on_disk.covariate) == {"cov_a", "cov_b"}, "checkpoint scope"
        print(f"[setup] checkpoint holds {n_all} rows over both covariates")

        # --- bug 1: resume targeting ONE of the two covariates ---
        sub = gps.permutation_significance(
            covariates=["cov_a"], B0=2, B1=2, random_seed=SEED,
            checkpoint_path=ck, resume=True, verbose=False)
        assert set(sub.covariate) == {"cov_a"}, "leaked an untested covariate"
        exp = full[full.covariate == "cov_a"].set_index(
            ["metabolite", "kernel_type"])["p_value"].sort_index()
        got = sub.set_index(["metabolite", "kernel_type"])["p_value"].sort_index()
        assert np.allclose(exp.to_numpy(), got.to_numpy()), \
            "subset resume changed p-values"
        print("[bug 1] covariate-subset resume: no crash, no leakage, "
              "p-values unchanged")

        # --- bug 2: resume targeting a SUBSET of outcomes ---
        keep = names[:2]
        g2 = build_fit()
        g2.models = {k: g2.models[k] for k in keep}
        g2.Y = g2.Y[keep]
        g2.out_names = keep
        few = g2.permutation_significance(
            covariates=["cov_a"], B0=2, B1=2, random_seed=SEED,
            checkpoint_path=ck, resume=True, verbose=False)
        assert set(few.metabolite) <= set(keep), "leaked an untested outcome"
        print("[bug 2] outcome-subset resume: no crash, no leakage")

        # --- bug 3: torn rows and out-of-scope rows counted separately ---
        with open(ck, "a") as f:
            f.write("y0,cov_a,lin")           # truncated final write
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gps.permutation_significance(
                covariates=["cov_a"], B0=2, B1=2, random_seed=SEED,
                checkpoint_path=ck, resume=True, verbose=True)
        msg = buf.getvalue()
        assert "1 truncated row(s) dropped" in msg, \
            f"torn row not reported distinctly:\n{msg}"
        assert "outside this run's scope ignored" in msg, \
            f"out-of-scope rows not reported distinctly:\n{msg}"
        print("[bug 3] truncated vs out-of-scope rows reported separately")

        # --- B1 derived from the screen (the DEFAULT path) ---
        # Every other case here passes B1 explicitly, so without this the
        # library's own default was untested.
        g3 = build_fit()
        ck3 = os.path.join(tmp, "derived.csv")
        der = g3.permutation_significance(
            covariates=["cov_a"], B0=2, q_target=0.5, B1_min=3, B1_max=6,
            random_seed=SEED, checkpoint_path=ck3, verbose=False)
        # count DRAWS, not rows: each draw emits one row per kernel type,
        # so .size() double-counts (6 draws x 2 kernels read as 12).
        n_per = (pd.read_csv(ck3).query("draw >= 0")
                 .groupby(["metabolite", "covariate"])["draw"].nunique())
        assert len(der), "derived-B1 run produced no results"
        assert n_per.max() <= 6, f"exceeded B1_max: {n_per.max()}"
        assert n_per.max() >= 3, f"below B1_min: {n_per.max()}"
        print(f"[derived B1] no explicit B1: drew up to {n_per.max()} per "
              f"component, within [B1_min=3, B1_max=6]")

        # --- the checkpoint is append-only: nothing was rewritten away ---
        after = pd.read_csv(ck)
        assert (after.covariate == "cov_b").sum() == \
            (on_disk.covariate == "cov_b").sum(), \
            "resume destroyed another covariate's draws"
        print("[invariant] untested covariate's draws still on disk")
        print("\nall checks passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
