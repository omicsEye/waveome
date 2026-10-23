"""run_search checkpoints each outcome's selected model and resumes from them.

A requeued simulation cell used to restart the kernel search from scratch,
losing days at the largest sizes. With checkpoint_dir set, each outcome's
selected model is saved as its search finishes; a rerun loads those and
searches only the rest. Checked here:

  1. a full run writes one checkpoint per outcome (outcome names with '/'
     included);
  2. a rerun loads a checkpointed outcome instead of searching it -- shown by
     planting a different model in its file and getting that model back;
  3. an outcome whose checkpoint is missing is searched again and selects
     exactly what the first run did.

No pytest in this environment, so this runs directly:

    python tests/test_search_checkpoint.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, "1")

import gpflow  # noqa: E402
from ray import cloudpickle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from waveome.model_search import GPSearch, _search_checkpoint_path  # noqa: E402

SEED = 9102
OUTCOMES = ["y1", "y/2"]


def make_data(n_units=12, n_visits=6):
    rng = np.random.default_rng(SEED)
    unit = np.repeat(np.arange(n_units), n_visits)
    t = np.tile(np.linspace(-1.5, 1.5, n_visits), n_units)
    u = np.repeat(rng.normal(0, 0.7, n_units), n_visits)
    X = pd.DataFrame({"id": unit.astype(float), "time": t})
    Y = pd.DataFrame({
        "y1": u + np.sin(2 * t) + rng.normal(0, 0.4, len(unit)),
        "y/2": 0.8 * t + rng.normal(0, 0.4, len(unit)),
    })
    return X, Y


def search(X, Y, checkpoint_dir):
    gps = GPSearch(X=X, Y=Y, unit_col="id", outcome_likelihood="gaussian")
    gps.run_search(kernels=[gpflow.kernels.SquaredExponential()],
                   max_depth=2, random_seed=SEED, num_jobs=2,
                   checkpoint_dir=checkpoint_dir)
    return gps.models


def params(model):
    return {k: np.asarray(v.numpy())
            for k, v in gpflow.utilities.parameter_dict(model).items()}


def same(a, b):
    pa, pb = params(a), params(b)
    return (a.kernel_name == b.kernel_name and pa.keys() == pb.keys()
            and all(np.array_equal(pa[k], pb[k]) for k in pa))


def main():
    X, Y = make_data()
    d = tempfile.mkdtemp()
    try:
        first = search(X, Y, d)
        paths = {o: _search_checkpoint_path(d, o) for o in OUTCOMES}
        missing = [o for o, p in paths.items() if not os.path.exists(p)]
        assert not missing, f"no checkpoint for {missing}"
        assert sorted(os.listdir(d)) == sorted(
            os.path.basename(p) for p in paths.values()), os.listdir(d)
        print(f"[ok] one checkpoint per outcome: {sorted(os.listdir(d))}")
        print(f"     first run: { {o: m.kernel_name for o, m in first.items()} }")

        # Plant y/2's model in y1's file, and drop y/2's file
        with open(paths["y1"], "wb") as handle:
            cloudpickle.dump(first["y/2"], handle)
        os.remove(paths["y/2"])

        second = search(X, Y, d)
        assert same(second["y1"], first["y/2"]), \
            "y1 was searched again instead of loaded from its checkpoint"
        print("[ok] checkpointed outcome loaded, not searched")
        assert same(second["y/2"], first["y/2"]), (
            f"re-searched y/2 differs: {second['y/2'].kernel_name} vs "
            f"{first['y/2'].kernel_name}")
        assert os.path.exists(paths["y/2"]), "re-searched outcome not saved"
        print("[ok] missing outcome searched again, identical to the first run")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
