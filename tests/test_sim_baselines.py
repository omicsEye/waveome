"""The simulation's NB GLM and GAM baselines select terms, and their refit is
checked.

The sweep's original GLM (per-unit effects, id x time, id x treat) had more
columns than rank, so it never produced standard errors and never selected
anything; its GAM could not test treat or treat x time. Checked here:

  1. a cell's data regenerate identically from its seed;
  2. the new GLM selects treat, time and their interaction when the data
     have them, with finite KL;
  3. the new GAM selects a curved time trend and time*treat when the data
     have them, and rarely a null term;
  4. a cell with every unit in one arm drops treat (GLM) or time*treat (GAM)
     instead of failing;
  5. the GAM predicts a finite holdout KL for a unit whose training counts
     are all zero and for a unit with no training rows;
  6. sim_baseline_refit.py refits finished cells, marks a cell whose stored
     value does not match its regenerated data as unverified, and resumes
     without adding rows; sim_aggregate.py --baseline-refit then uses only
     verified rows.

No pytest in this environment, so this runs directly:

    python tests/test_sim_baselines.py
"""
import os
import pickle
import subprocess
import sys
import tempfile
import warnings

# One BLAS thread, as the refit script sets, before numpy is imported here
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "examples", "simulations")
sys.path.insert(0, SIM)
import sim_aggregate as agg  # noqa: E402
import sim_baseline_refit as refit  # noqa: E402
import sim_waveome_hpc_run as h  # noqa: E402

warnings.filterwarnings("ignore")
CELL = dict(rate=4.0, units=10, epsilon=0.0, alpha=10.0)


def nb_data(n_units=40, rate=8, treat_effect=1.0, time_effect=0.8, curve=0.0,
            all_control=False, seed=9102):
    """Counts from an NB GLM with a unit intercept, a treatment shift, a
    treatment-specific time slope and a time curve shared by both arms, with
    the dicts the baselines' KL needs."""
    rng = np.random.default_rng(seed)
    treat = np.zeros(n_units) if all_control else rng.integers(0, 2, n_units)
    rows = []
    for u in range(n_units):
        for t in rng.uniform(-1.5, 1.5, rate):
            rows.append((u, treat[u], t))
    df = pd.DataFrame(rows, columns=["id", "treat", "time"])
    unit = rng.normal(0, 0.3, n_units)[df.id]
    mu = np.exp(1 + unit + treat_effect * df.treat + time_effect * df.treat * df.time
                + curve * (df.time ** 2 - 0.75))
    alpha = 0.5
    n, p = 1 / alpha, 1 / (1 + alpha * mu)
    df["y1"] = rng.negative_binomial(n, p).astype(float)
    true = {"y1": {"n": np.full(len(df), n), "p": p.to_numpy()}}
    idx = rng.permutation(len(df))
    tr, ho = idx[: int(0.8 * len(df))], idx[int(0.8 * len(df)):]
    cut = lambda d, ix: {"y1": {k: v[ix] for k, v in d["y1"].items()}}
    return df.iloc[tr], df.iloc[ho], cut(true, tr), cut(true, ho)


def check_regeneration():
    a = h.simulate_split(h.kern_out, CELL["rate"], CELL["units"], CELL["epsilon"], CELL["alpha"], 3)
    b = h.simulate_split(h.kern_out, CELL["rate"], CELL["units"], CELL["epsilon"], CELL["alpha"], 3)
    pd.testing.assert_frame_equal(a[1], b[1])
    pd.testing.assert_frame_equal(a[2], b[2])
    assert np.array_equal(a[4]["y3"]["p"], b[4]["y3"]["p"])
    print("[ok] a cell's data and split regenerate identically from its seed")


def check_glm_selects():
    tr, ho, td, hd = nb_data()
    out = h.fit_nb_glm("y1", tr, ho, td, hd)
    assert out["status"] == "ok", out
    assert set(out["features"].split(",")) == {"treat", "time*treat"}, out["features"]
    assert np.isfinite([out[k] for k in ["train_kl", "holdout_kl", "train_mse", "holdout_mse"]]).all()
    # Null: each term is called at most a little above 5% of the time (with
    # 40 units, clustered errors run ~7%)
    calls = [h.fit_nb_glm("y1", *nb_data(treat_effect=0, time_effect=0, seed=s))["features"]
             for s in range(60)]
    for term in ["time", "treat", "time*treat"]:
        rate = np.mean([term in c.split(",") for c in calls])
        assert rate <= 0.15, f"{term} called in {rate:.0%} of null datasets"
    print("[ok] the GLM selects the built-in treat and time*treat terms; null terms rarely")


def check_gam_selects():
    tr, ho, td, hd = nb_data(curve=0.6)
    out = h.fit_nb_gam("y1", tr, ho, td, hd)
    assert out["status"] == "ok", out
    assert set(out["features"].split(",")) == {"time", "time*treat"}, out["features"]
    assert np.isfinite([out[k] for k in ["train_kl", "holdout_kl", "train_mse", "holdout_mse"]]).all()
    # Null: the unit intercepts absorb the unit effects, so null terms are
    # called at most ~5% of the time (~1% in these data)
    calls = [h.fit_nb_gam("y1", *nb_data(treat_effect=0, time_effect=0, seed=s))["features"]
             for s in range(60)]
    for term in ["time", "time*treat"]:
        rate = np.mean([term in c.split(",") for c in calls])
        assert rate <= 0.1, f"{term} called in {rate:.0%} of null datasets"
    print("[ok] the GAM selects a curved time trend and time*treat; null terms rarely")


def check_one_arm():
    tr, ho, td, hd = nb_data(all_control=True, time_effect=0)
    out = h.fit_nb_glm("y1", tr, ho, td, hd)
    assert out["status"] == "ok" and "treat" not in out["features"], out
    assert np.isfinite(out["holdout_kl"])
    out = h.fit_nb_gam("y1", tr, ho, td, hd)
    assert out["status"] == "ok" and "treat" not in out["features"], out
    assert np.isfinite(out["holdout_kl"])
    print("[ok] a cell with every unit in one arm drops treat and still fits")


def check_gam_sparse_units():
    tr, ho, td, hd = nb_data(curve=0.6)
    zero_unit = ho.id.iloc[0]  # a unit with holdout rows
    tr = tr.assign(y1=tr.y1.where(tr.id != zero_unit, 0.0))
    # Copies of one unit's holdout rows, as a unit with no training rows
    copied = (ho.id == ho.id.iloc[-1]).to_numpy()
    ho = pd.concat([ho, ho[copied].assign(id=tr.id.max() + 1)])
    hd = {"y1": {k: np.concatenate([v, v[copied]]) for k, v in hd["y1"].items()}}
    out = h.fit_nb_gam("y1", tr, ho, td, hd)
    assert out["status"] == "ok", out
    assert np.isfinite([out[k] for k in ["train_kl", "holdout_kl", "train_mse", "holdout_mse"]]).all(), out
    print("[ok] the GAM predicts units with all-zero training counts and units without training rows")


def write_cell(cells, run_id, stored_kl, n_train):
    met = pd.DataFrame([dict(model=m, output=o, features="", train_kl=0.1, holdout_kl=0.1,
                             train_mse=1.0, holdout_mse=1.0)
                        for m in ["glm", "gam", "lasso"] for o in ["y1", "y2", "y3", "y4"]])
    met.loc[(met.model == "glm") & (met.output == "y1"), "holdout_kl"] = stored_kl
    res = pd.DataFrame({"stage": ["nb_glm"], "wall_s": [1.0], "peak_rss_gb": [1.0], "n_train": n_train})
    name = "cell_rate{rate:g}_units{units}_eps{epsilon:g}_alpha{alpha:g}".format(**CELL)
    with open(os.path.join(cells, f"{name}_run{run_id}.pickle"), "wb") as handle:
        pickle.dump({"metrics": met, "permutation_draws": None, "resources": res,
                     "provenance": {"git_commit": "abc1234"}}, handle)
    return met.assign(**CELL, run_id=run_id)


def check_refit_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        cells = os.path.join(tmp, "cells")
        os.makedirs(cells)
        metrics = []
        for run_id, corrupt in [(3, False), (4, True)]:
            s, tr, ho, td, hd = h.simulate_split(
                h.kern_out, CELL["rate"], CELL["units"], CELL["epsilon"], CELL["alpha"], run_id)
            kl = refit.original_glm_holdout_kl("y1", tr, ho, hd, sorted(set(s.id)))
            metrics.append(write_cell(cells, run_id, kl + 1 if corrupt else kl, len(tr)))
        with open(os.path.join(cells, "cell_rate4_units10_eps0_alpha10_run5.pickle"), "wb") as handle:
            pickle.dump({"metrics": pd.DataFrame(), "permutation_draws": None, "resources": None}, handle)

        out = os.path.join(tmp, "baseline_refit.csv")
        cmd = [sys.executable, os.path.join(SIM, "sim_baseline_refit.py"), "--cells", cells, "--out", out]
        p = subprocess.run(cmd, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-2000:]
        rows = pd.read_csv(out)
        assert len(rows) == 16 and set(rows.run_id) == {3, 4}
        assert rows.groupby("model").size().to_dict() == {"gam": 8, "glm": 8}
        assert rows.groupby("run_id").verified.first().to_dict() == {3: True, 4: False}
        assert set(rows.status) <= {"ok", "no_fit", "wald_failed"}

        p = subprocess.run(cmd, capture_output=True, text=True)
        assert p.returncode == 0 and "0 cells to refit" in p.stdout, p.stdout
        assert len(pd.read_csv(out)) == 16, "resume added rows"

        swapped, note = agg.use_baseline_refit(pd.concat(metrics, ignore_index=True), rows)
        for model in ["glm", "gam"]:
            new = swapped[swapped.model == model]
            assert set(new.run_id) == {3} and len(new) == 4, model
        assert (swapped.model == "lasso").sum() == 8
        assert "1 cells have no GLM or GAM rows (1 failed the data check)" in note, note
    print("[ok] refit marks a mismatched cell unverified, resumes, and aggregation uses verified rows")


def main():
    check_regeneration()
    check_glm_selects()
    check_gam_selects()
    check_one_arm()
    check_gam_sparse_units()
    check_refit_end_to_end()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
