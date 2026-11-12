"""The simulation aggregation turns cells into the right calls and scores.

Synthetic inputs with known answers:

  1. term spellings are normalized (time*treat == treat*time, time*time == time);
  2. a baseline's selected terms give the expected covariate and term calls;
  3. duplicate and non-finite permutation draws are dropped and counted;
  4. pooled permutation tests call exactly the components built to be
     significant, an interaction only when it rejects under both covariates,
     and a test with a non-finite observed statistic is not called;
  5. confusion counts and F1 / FDR match a hand count;
  6. the whole script runs end to end on synthetic cell pickles, including
     an empty cell.

No pytest in this environment, so this runs directly:

    python tests/test_sim_aggregate.py
"""
import os
import pickle
import subprocess
import sys
import tempfile

# One BLAS thread, as sim_aggregate.py sets, before numpy is imported here
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "examples", "simulations")
sys.path.insert(0, SIM)
import sim_aggregate as agg  # noqa: E402

SETTING = dict(rate=4.0, units=10, epsilon=0.0, alpha=0.5)
# Penalized candidate components in kernel order, as the harness records them
CANDIDATES = "id,treat,time,treat*time"
STRONG = {  # (outcome, covariate, kernel_idx) built to be far beyond the null
    ("y1", "time", 2),
    ("y3", "treat", 3), ("y3", "time", 3),   # interaction under both
    ("y4", "time", 3),                        # interaction under time only
}
TESTED = [("treat", 1, "categorical"), ("treat", 3, "categorical*squared_exponential"),
          ("time", 2, "squared_exponential"), ("time", 3, "categorical*squared_exponential")]


def make_draws(run_id, rng, n_null=100):
    rows = []
    for o in ["y1", "y2", "y3", "y4"]:
        for cov, idx, kt in TESTED:
            base = dict(metabolite=o, covariate=cov, kernel_type=kt, kernel_idx=idx)
            obs = 40.0 if (o, cov, idx) in STRONG else 0.0  # 0 = the null median
            rows.append({**base, "draw": -1, "log_bf": obs})
            for b, v in enumerate(rng.normal(size=n_null)):
                rows.append({**base, "draw": b, "log_bf": v})
    return pd.DataFrame(rows).assign(**SETTING, run_id=run_id)


def make_metrics(run_id, features):
    rows = []
    for o in ["y1", "y2", "y3", "y4"]:
        rows.append(dict(model="waveome_penalized", output=o, features=CANDIDATES))
        for method, f in features.items():
            rows.append(dict(model=method, output=o, features=f[o]))
    return pd.DataFrame(rows).assign(**SETTING, run_id=run_id,
                                     train_kl=0.1, holdout_kl=0.1)


def check_norm_and_baseline_calls():
    assert agg.norm_term("time*treat") == "treat*time"
    assert agg.norm_term("time*time") == "time"
    assert agg.norm_term("time*id") == "id*time"
    m = make_metrics(0, {"lasso": {"y1": "id,time*id", "y2": "treat", "y3": "time*treat",
                                   "y4": ""}})
    calls = {r["output"]: r for r in agg.method_calls(m)}
    assert calls["y1"]["cov_called"] == {"time"} and calls["y1"]["term_called"] == set()
    assert calls["y2"]["cov_called"] == {"treat"} and calls["y2"]["term_called"] == {"treat"}
    assert calls["y3"]["cov_called"] == {"treat", "time"}
    assert calls["y3"]["term_called"] == {"treat*time"}
    assert calls["y4"]["cov_called"] == set() and calls["y4"]["term_called"] == set()
    print("[ok] term spellings normalized; baseline calls from selected terms")


def check_clean_draws():
    d = make_draws(0, np.random.default_rng(9102))
    n = len(d)
    d.loc[5, "log_bf"] = np.nan
    d.loc[6, "log_bf"] = np.inf
    d = pd.concat([d, d.iloc[[7, 8]]], ignore_index=True)  # two duplicates
    clean, counts = agg.clean_draws(d)
    assert len(clean) == n - 2
    assert counts.n_duplicate.sum() == 2 and counts.n_nonfinite.sum() == 2
    assert np.isfinite(clean.log_bf).all()
    print("[ok] duplicate and non-finite draws dropped and counted")


def check_penalized_calls():
    rng = np.random.default_rng(9102)
    draws = pd.concat([make_draws(r, rng) for r in range(20)], ignore_index=True)
    # Replicate 0's y1 time statistic is non-finite: dropped, so no test, no call
    nan_row = ((draws.run_id == 0) & (draws.metabolite == "y1")
               & (draws.covariate == "time") & (draws.kernel_idx == 2) & (draws.draw == -1))
    draws.loc[nan_row, "log_bf"] = np.nan
    draws, _ = agg.clean_draws(draws)
    tests = agg.penalized_tests(draws, q=0.10)
    assert len(tests) == 20 * 4 * 4 - 1
    built = tests.apply(lambda r: (r.metabolite, r.covariate, r.kernel_idx) in STRONG, axis=1)
    assert tests.hit[built].all() and not tests.hit[~built].any()
    metrics = pd.concat([make_metrics(r, {}) for r in range(20)], ignore_index=True)
    calls = {(c["run_id"], c["output"]): c for c in agg.penalized_calls(tests, metrics)}
    assert calls[(0, "y1")]["cov_called"] == set()
    assert calls[(1, "y1")]["cov_called"] == {"time"}
    assert calls[(1, "y1")]["term_called"] == {"time"}
    assert calls[(1, "y3")]["term_called"] == {"treat*time"}  # both permutations
    assert calls[(1, "y3")]["cov_called"] == {"treat", "time"}
    assert calls[(1, "y4")]["term_called"] == set()            # time permutation only
    assert calls[(1, "y4")]["cov_called"] == {"time"}
    assert calls[(1, "y2")]["cov_called"] == set()
    print("[ok] pooled tests call exactly the built-in signals; interaction needs both")


def check_scores():
    calls = pd.DataFrame([
        dict(**SETTING, run_id=0, method="m", output="y1", cov_called={"time", "treat"},
             term_called={"time"}),
        dict(**SETTING, run_id=0, method="m", output="y3", cov_called={"time"},
             term_called=set()),
    ])
    s = agg.score(calls)
    cov = s[(s.level == "covariate") & (s.output == "all")].iloc[0]
    # y1: time TP, treat FP. y3: time TP, treat FN.
    assert (cov.tp, cov.fp, cov.fn, cov.tn) == (2, 1, 1, 0)
    assert np.isclose(cov.f1, 2 * 2 / (2 * 2 + 1 + 1)) and np.isclose(cov.fdr, 1 / 3)
    term = s[(s.level == "term") & (s.output == "all")].iloc[0]
    # y1: time TP, treat TN, treat*time TN. y3: treat*time FN, treat TN, time TN.
    assert (term.tp, term.fp, term.fn, term.tn) == (1, 0, 1, 4)
    print("[ok] confusion counts, F1 and FDR match a hand count")


def check_end_to_end():
    rng = np.random.default_rng(9102)
    feats = {"glm": {o: "time" for o in ["y1", "y2", "y3", "y4"]}}
    with tempfile.TemporaryDirectory() as tmp:
        cells = os.path.join(tmp, "cells")
        os.makedirs(cells)
        for r in range(12):
            # As the harness writes it: one row per stage, n_train repeated
            res = pd.DataFrame({"stage": ["penalized", "permutation"],
                                "wall_s": [1800.0, 1800.0], "peak_rss_gb": [1.5, 2.0],
                                "n_train": 32})
            with open(os.path.join(cells, f"cell_rate4_units10_eps0_alpha0.5_run{r}.pickle"), "wb") as h:
                pickle.dump({"metrics": make_metrics(r, feats).drop(columns=list(SETTING) + ["run_id"]),
                             "permutation_draws": make_draws(r, rng).drop(columns=list(SETTING) + ["run_id"]),
                             "resources": res, "provenance": {"git_commit": "abc1234def"}}, h)
        with open(os.path.join(cells, "cell_rate4_units10_eps0_alpha0.5_run12.pickle"), "wb") as h:
            pickle.dump({"metrics": pd.DataFrame(), "permutation_draws": None,
                         "resources": None, "provenance": {"git_commit": "abc1234def"}}, h)
        out = os.path.join(tmp, "agg")
        p = subprocess.run([sys.executable, os.path.join(SIM, "sim_aggregate.py"),
                            "--cells", cells, "--out-dir", out],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        status = pd.read_csv(os.path.join(out, "cell_status.csv"))
        assert (status.status == "ok").sum() == 12 and (status.status == "empty").sum() == 1
        ok = status[status.status == "ok"]
        assert (ok.n_train == 32).all() and np.allclose(ok.hours, 1.0)
        assert np.allclose(ok.peak_gb, 2.0) and (ok.commit == "abc1234").all()
        s = pd.read_csv(os.path.join(out, "scores.csv"))
        glm = s[(s.method == "glm") & (s.level == "covariate") & (s.output == "all")].iloc[0]
        # glm calls time everywhere (4 TP per replicate) and misses treat on y3, y4
        assert (glm.tp, glm.fp, glm.fn, glm.tn) == (48, 0, 24, 24)
        pen = s[(s.method == "waveome_penalized") & (s.level == "covariate")
                & (s.output == "all")].iloc[0]
        assert (pen.tp, pen.fp, pen.fn, pen.tn) == (12 * 4, 0, 12 * 2, 12 * 2)
    print("[ok] end to end on synthetic cell pickles, empty cell reported")


def main():
    check_norm_and_baseline_calls()
    check_clean_draws()
    check_penalized_calls()
    check_scores()
    check_end_to_end()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
