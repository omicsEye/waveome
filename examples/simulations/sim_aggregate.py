"""Aggregate the simulation sweep's cells into per-method calls and scores.

Reads every cell pickle written by sim_waveome_hpc_run.py and writes, to
--out-dir:

  cell_status.csv     one row per cell: ok or empty (the data simulation
                      failed), code commit, training size, run time, peak
                      memory, and how many permutation draws were dropped
  metrics.csv         every cell's per-method, per-outcome rows (features,
                      KL, MSE), for the KL / MSE figures
  penalized_tests.csv the penalized variant's permutation tests: p-value,
                      BH q-value and hit, per replicate, outcome and component
  calls.csv           covariates and terms each method called, per replicate
                      and outcome
  scores.csv          TP / FP / FN / TN, sensitivity, specificity, precision,
                      F1 and realized FDR, per setting, method, level and
                      outcome (outcome "all" pools the four)

Penalized variant (author's decisions, IMPLEMENTATION_PLAN frozen decision 5):
draws are pooled over all replicates of a setting, one family per
(covariate, kernel type) stratum; p-values come from calc_permutation_pvalues
and BH is applied at --q within each family. A covariate is called when any
of its components rejects; an interaction term is called only when it
rejects under the permutation of every covariate it contains. Every other
method's calls are the terms it selected (the `features` column).

Scoring levels: covariate (treat, time; primary) and term (treat, time,
treat*time; supplement). `id` is an adjuster and is not scored, so terms
involving it count only at the covariate level, through their other factor.

Permutation draws are cleaned first: duplicate draws (a resumed cell reruns
draws whose value was NaN and appends them again) are removed, and a draw
with any non-finite component is dropped whole, as the library now treats
it like a draw that raised an error (TODO A2). A test whose observed
statistic is non-finite has no p-value and is not called.

Usage, from examples/simulations (on the cluster, or on a copy of cells/):
    python sim_aggregate.py --cells ./sim_waveome_output/cells \\
        --out-dir ./sim_waveome_output/aggregate
"""
import argparse
import glob
import os
import pickle
import re

# One BLAS thread: the p-values come from many small quantile regressions, and
# multithreaded BLAS made them ~10-20x slower. Set before numpy is imported;
# parallelism comes from --jobs instead.
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from waveome.utilities import calc_bh_qvalues, calc_permutation_pvalues  # noqa: E402

SETTING = ["rate", "units", "epsilon", "alpha"]
COVS = ["treat", "time"]
TERMS = ["treat", "time", "treat*time"]
# True terms per outcome, read off the kernels in sim_waveome_hpc_run.py
# (k1-k4), leaving out `id` terms. Covariate truth is every covariate the
# true terms contain.
TERM_TRUTH = {"y1": {"time"}, "y2": {"time"},
              "y3": {"treat*time"}, "y4": {"treat*time"}}
COV_TRUTH = {o: {c for t in ts for c in t.split("*")}
             for o, ts in TERM_TRUTH.items()}
FACTOR_ORDER = ["id", "treat", "time"]
PENALIZED = "waveome_penalized"
CELL_RE = re.compile(r"cell_rate([\d.]+)_units(\d+)_eps([\d.]+)_alpha([\d.]+)_run(\d+)\.pickle$")
DRAW_KEY = SETTING + ["run_id", "metabolite", "covariate", "kernel_idx", "draw"]


def norm_term(term):
    """One spelling per term: factors deduplicated and in a fixed order."""
    parts = dict.fromkeys(p.strip() for p in term.split("*") if p.strip())
    return "*".join(sorted(parts, key=lambda p: (
        FACTOR_ORDER.index(p) if p in FACTOR_ORDER else len(FACTOR_ORDER), p)))


def split_features(features):
    if not isinstance(features, str):
        return []
    return [norm_term(t) for t in features.split(",") if t.strip()]


def load_cells(cell_dir):
    """Status, metrics and raw permutation draws from every cell pickle."""
    status, metrics, draws = [], [], []
    for path in sorted(glob.glob(os.path.join(cell_dir, "cell_*.pickle"))):
        m = CELL_RE.search(path)
        if m is None:
            continue
        setting = dict(zip(SETTING + ["run_id"],
                           [float(m[1]), int(m[2]), float(m[3]), float(m[4]), int(m[5])]))
        with open(path, "rb") as handle:
            d = pickle.load(handle)
        row = {**setting, "file": os.path.basename(path),
               "commit": (d.get("provenance", {}).get("git_commit") or "")[:7]}
        res = d.get("resources")
        if res is None or d.get("metrics") is None or not len(d["metrics"]):
            status.append({**row, "status": "empty"})
            continue
        r = pd.DataFrame(res)
        status.append({**row, "status": "ok", "n_train": int(r["n_train"].iloc[0]),
                       "hours": r["wall_s"].sum() / 3600,
                       "peak_gb": r["peak_rss_gb"].max()})
        metrics.append(d["metrics"].assign(**setting))
        if d.get("permutation_draws") is not None:
            draws.append(d["permutation_draws"].assign(**setting))
    status = pd.DataFrame(status)
    metrics = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    draws = pd.concat(draws, ignore_index=True) if draws else pd.DataFrame(columns=DRAW_KEY + ["kernel_type", "log_bf"])
    return status, metrics, draws


def clean_draws(draws):
    """Drop duplicate and non-finite draws; return the rest and per-cell counts.

    A draw with any non-finite component is dropped whole (every component
    of that outcome, covariate and draw), as the library now fails such a
    draw (TODO A2); cells run before that fix still hold its finite rows.
    """
    dup = draws.duplicated(DRAW_KEY)
    draw = [k for k in DRAW_KEY if k != "kernel_idx"]
    nonfinite = (~np.isfinite(draws["log_bf"].astype(float))).groupby(
        [draws[k] for k in draw]).transform("any")
    bad = nonfinite & ~dup
    cell = SETTING + ["run_id"]
    counts = (draws.assign(n_draws=1, n_duplicate=dup, n_nonfinite=bad)
              .groupby(cell)[["n_draws", "n_duplicate", "n_nonfinite"]].sum()
              .astype(int).reset_index())
    return draws[~dup & ~bad].reset_index(drop=True), counts


def _family_tests(key, g):
    """P-values and BH q-values for one family (setting x stratum)."""
    test = ["run_id", "metabolite", "kernel_idx"]
    obs = {tuple(k): v for *k, v in
           g[g.draw == -1][test + ["log_bf"]].itertuples(index=False)}
    nul = {k: v["log_bf"].to_numpy() for k, v in g[g.draw >= 0].groupby(test)}
    obs = {k: v for k, v in obs.items() if k in nul}
    if not obs:
        return None
    p = calc_permutation_pvalues(obs, nul)
    p["q_value"] = calc_bh_qvalues(p["p_value"].to_numpy())
    p.index = pd.MultiIndex.from_tuples(p.index, names=test)
    p = p.reset_index()
    p["log_bf"] = [obs[k] for k in zip(p.run_id, p.metabolite, p.kernel_idx)]
    for col, val in zip(SETTING + ["covariate", "kernel_type"], key):
        p[col] = val
    return p


def penalized_tests(draws, q, jobs=1):
    """P-values and BH calls, one family per setting and stratum."""
    out = Parallel(n_jobs=jobs)(
        delayed(_family_tests)(k, g)
        for k, g in draws.groupby(SETTING + ["covariate", "kernel_type"]))
    out = [p for p in out if p is not None]
    if not out:
        return pd.DataFrame(columns=SETTING + ["run_id", "metabolite", "kernel_idx",
                                               "covariate", "kernel_type", "p_value",
                                               "q_value", "hit"])
    res = pd.concat(out, ignore_index=True)
    res["hit"] = res["q_value"] <= q
    return res


def penalized_calls(tests, metrics):
    """Covariate and term calls of the penalized variant per replicate and outcome."""
    key = SETTING + ["run_id"]
    hits = {k: {(r.covariate, int(r.kernel_idx)): bool(r.hit)
                for r in g.itertuples(index=False)}
            for k, g in tests.groupby(key + ["metabolite"])}
    rows = []
    for m in metrics[metrics.model == PENALIZED].itertuples(index=False):
        cell = tuple(getattr(m, k) for k in key) + (m.output,)
        terms = split_features(m.features)  # candidate components, kernel order
        hit = hits.get(cell, {})
        cov_called = {c for (c, _), h in hit.items() if h}
        term_called = set()
        for idx, term in enumerate(terms):
            covs = [f for f in term.split("*") if f in COVS]
            if term in TERMS and covs and all(hit.get((c, idx), False) for c in covs):
                term_called.add(term)
        rows.append(dict(zip(SETTING + ["run_id", "output"], cell), method=PENALIZED,
                         cov_called=cov_called, term_called=term_called))
    return rows


def method_calls(metrics):
    """Calls of every method other than the penalized variant (its selected terms)."""
    rows = []
    for m in metrics[metrics.model != PENALIZED].itertuples(index=False):
        terms = set(split_features(m.features))
        rows.append({**{k: getattr(m, k) for k in SETTING + ["run_id", "output"]},
                     "method": m.model,
                     "cov_called": {c for c in COVS if any(c in t.split("*") for t in terms)},
                     "term_called": terms & set(TERMS)})
    return rows


def score(calls):
    """Confusion counts and rates per setting, method, level and outcome."""
    rows = []
    for level, cands, truth, col in [("covariate", COVS, COV_TRUTH, "cov_called"),
                                     ("term", TERMS, TERM_TRUTH, "term_called")]:
        for c in cands:
            t = calls.output.map(lambda o: c in truth[o])
            k = calls[col].map(lambda s: c in s)
            rows.append(calls[SETTING + ["method", "output"]].assign(
                level=level, tp=t & k, fp=~t & k, fn=t & ~k, tn=~t & ~k))
    d = pd.concat(rows, ignore_index=True)
    by = SETTING + ["method", "level"]
    counts = pd.concat([
        d.groupby(by + ["output"])[["tp", "fp", "fn", "tn"]].sum().reset_index(),
        d.groupby(by)[["tp", "fp", "fn", "tn"]].sum().reset_index().assign(output="all"),
    ], ignore_index=True)
    counts["n_replicates"] = counts.merge(
        calls.groupby(SETTING + ["method"]).run_id.nunique().rename("n").reset_index(),
        on=SETTING + ["method"], how="left")["n"].to_numpy()
    tp, fp, fn, tn = (counts[c].astype(int) for c in ["tp", "fp", "fn", "tn"])
    counts["sensitivity"] = tp / (tp + fn).clip(lower=1)
    counts["specificity"] = tn / (tn + fp).clip(lower=1)
    counts["precision"] = tp / (tp + fp).clip(lower=1)
    counts["f1"] = 2 * tp / (2 * tp + fp + fn).clip(lower=1)
    # Realized FDR: false calls over all calls, 0 when nothing is called
    counts["fdr"] = fp / (tp + fp).clip(lower=1)
    # Covariate level: treat is absent from y1 and y2 only, and time is in every
    # outcome's truth, so its specificity rests on those two outcomes' treat calls
    return counts.sort_values(by + ["output"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", default="./sim_waveome_output/cells")
    ap.add_argument("--out-dir", default="./sim_waveome_output/aggregate")
    ap.add_argument("--q", type=float, default=0.10, help="BH target")
    ap.add_argument("--jobs", type=int, default=1,
                    help="families (setting x stratum) tested in parallel")
    args = ap.parse_args()

    status, metrics, draws = load_cells(args.cells)
    if not len(status):
        raise SystemExit(f"no cell pickles in {args.cells}")
    draws, counts = clean_draws(draws)
    status = status.merge(counts, on=SETTING + ["run_id"], how="left")
    tests = penalized_tests(draws, args.q, args.jobs)
    calls = pd.DataFrame(penalized_calls(tests, metrics) + method_calls(metrics))
    scores = score(calls)

    os.makedirs(args.out_dir, exist_ok=True)
    status.to_csv(os.path.join(args.out_dir, "cell_status.csv"), index=False)
    metrics.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
    tests.to_csv(os.path.join(args.out_dir, "penalized_tests.csv"), index=False)
    calls.assign(**{c: calls[c].map(lambda s: ",".join(sorted(s)))
                    for c in ["cov_called", "term_called"]}
                 ).to_csv(os.path.join(args.out_dir, "calls.csv"), index=False)
    scores.to_csv(os.path.join(args.out_dir, "scores.csv"), index=False)

    pd.set_option("display.width", 200)
    ok = status[status.status == "ok"]
    print(f"cells: {len(ok)} ok, {int((status.status == 'empty').sum())} empty "
          f"(data simulation failed), over {status.groupby(SETTING).ngroups} settings")
    print(f"commits: {status.commit.value_counts().to_dict()}")
    print(f"draws dropped: {int(status.n_duplicate.sum())} duplicate, "
          f"{int(status.n_nonfinite.sum())} non-finite, "
          f"in {int((status.n_nonfinite > 0).sum())} cells")
    n_obs = int((draws.draw == -1).sum()) if len(draws) else 0
    print(f"penalized tests: {len(tests)} with a p-value "
          f"({n_obs - len(tests)} observed statistics without a usable null)")
    pooled = scores[scores.output == "all"]
    summ = (pooled.groupby(["level", "method"])[["tp", "fp", "fn", "tn"]].sum())
    tp, fp, fn, tn = (summ[c] for c in ["tp", "fp", "fn", "tn"])
    summ["sens"] = tp / (tp + fn).clip(lower=1)
    summ["spec"] = tn / (tn + fp).clip(lower=1)
    summ["f1"] = 2 * tp / (2 * tp + fp + fn).clip(lower=1)
    summ["fdr"] = fp / (tp + fp).clip(lower=1)
    print("\nAll settings pooled (per-setting rows are in scores.csv):")
    print(summ.round(3).to_string())
    print(f"\nwrote {args.out_dir}/")


if __name__ == "__main__":
    main()
