"""Refit the NB GLM and GAM baselines on finished sweep cells.

The sweep's original GLM never selected anything, and its GAM could not test
treat or treat x time (see fit_nb_glm and fit_nb_gam in
sim_waveome_hpc_run.py). This regenerates each finished cell's data from its
seed, refits both with fit_nb_glm and fit_nb_gam, and appends the rows to
--out, which sim_aggregate.py --baseline-refit uses in place of the cells'
GLM and GAM rows.

Each cell's regenerated data are checked against the cell: the training size
must match, and the original GLM's y1 holdout KL, recomputed on the
regenerated data with the original code, must match the value the cell
stored. The outcomes come from a multivariate normal draw that could differ
across machines, and the training size alone would not show it. Rows of a
cell that fails the check are written with verified=False and not used.

Resumable: cells already in --out are skipped. Run it after the sweep has
finished, from examples/simulations. A 500-unit, rate-16 cell takes ~10
minutes and ~8 GB (most of it regenerating the data), e.g.
    srun -p cpu -c 8 --mem=72G -t 24:00:00 python sim_baseline_refit.py --jobs 8
"""
import argparse
import os

# One BLAS thread, as in the sweep (sim_waveome_hpc_run.sh); parallelism
# comes from --jobs. Set before numpy is imported.
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import glob  # noqa: E402
import pickle  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels as sm  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from statsmodels.formula.api import glm  # noqa: E402

import sim_waveome_hpc_run as h  # noqa: E402
from sim_aggregate import CELL_RE, SETTING  # noqa: E402

OUTPUTS = ["y1", "y2", "y3", "y4"]
METRICS = ["train_kl", "holdout_kl", "train_mse", "holdout_mse"]


def original_glm_holdout_kl(o, train_df, holdout_df, holdout_dict, id_list):
    """Holdout KL of the sweep's original GLM for outcome `o`, copied from the
    block fit_nb_glm replaced (for the first outcome, whose fallback did not
    depend on an earlier outcome's model)."""
    best_nb, best_nb_ll, best_alpha = None, -np.inf, None
    for a in np.arange(start=1, stop=11):
        m_nb_glm = glm(
            formula=f"{o} ~ C(id, levels=id_list) + time + treat + C(id, levels=id_list) * time + C(id, levels=id_list) * treat + time * treat",
            data=train_df,
            family=sm.genmod.families.NegativeBinomial(alpha=a)
        )
        try:
            m_nb_glm = m_nb_glm.fit(cov_type="cluster", cov_kwds={"groups": train_df.id})
        except Exception:
            try:
                m_nb_glm = m_nb_glm.fit(method="lbfgs")
            except Exception:
                continue
        if m_nb_glm.llf > best_nb_ll:
            best_nb, best_nb_ll, best_alpha = m_nb_glm, m_nb_glm.llf, a

    final = best_nb
    if best_nb is not None:
        try:
            sig = [x for x in best_nb.wald_test_terms(scalar=True).table.query("pvalue < 0.05").index
                   if x != "Intercept"]
            refit = glm(
                formula=f"{o} ~ {'+'.join(sig) if sig else '1'}",
                data=train_df,
                family=sm.genmod.families.NegativeBinomial(alpha=best_alpha)
            )
            try:
                final = refit.fit(cov_type="cluster", cov_kwds={"groups": train_df.id})
            except Exception:
                try:
                    final = refit.fit()
                except Exception:
                    final = best_nb
        except Exception:
            final = best_nb
    try:
        hat = final.predict(holdout_df)
    except Exception:
        hat = np.repeat(train_df[o].mean(), repeats=holdout_df.shape[0])
    return h.calc_kl_all(
        n=holdout_dict[o]["n"], p=holdout_dict[o]["p"],
        x=holdout_df[["id", "treat", "time"]], y=holdout_df[o],
        est_m=None, m_type="glm", y_pred=hat
    )


def refit_cell(setting, stored_n_train, stored_kl):
    """New GLM and GAM rows for one cell, with the data check."""
    warnings.filterwarnings("ignore")
    sim_df, train_df, holdout_df, train_dict, holdout_dict = h.simulate_split(
        h.kern_out, setting["rate"], setting["units"], setting["epsilon"],
        setting["alpha"], setting["run_id"]
    )
    if sim_df is None:
        return pd.DataFrame()
    id_list = sorted(set(sim_df.id))
    regen_kl = original_glm_holdout_kl("y1", train_df, holdout_df, holdout_dict, id_list)
    verified = bool(len(train_df) == stored_n_train and (
        np.isclose(regen_kl, stored_kl, rtol=1e-6) or (np.isnan(regen_kl) and np.isnan(stored_kl))))
    rows = [{"output": o, "model": model,
             **fit(o, train_df, holdout_df, train_dict, holdout_dict)}
            for model, fit in [("glm", h.fit_nb_glm), ("gam", h.fit_nb_gam)]
            for o in OUTPUTS]
    return pd.DataFrame(rows).assign(
        **setting, n_train=len(train_df), verified=verified,
        stored_y1_kl=stored_kl, regen_y1_kl=regen_kl)


def finished_cells(cell_dir):
    """(setting, stored training size, stored original GLM y1 holdout KL)
    for every cell pickle that has results."""
    cells = []
    for path in sorted(glob.glob(os.path.join(cell_dir, "cell_*.pickle"))):
        m = CELL_RE.search(path)
        if m is None:
            continue
        with open(path, "rb") as handle:
            d = pickle.load(handle)
        if d.get("resources") is None or d.get("metrics") is None or not len(d["metrics"]):
            continue
        met = d["metrics"]
        kl = met.loc[(met.model == "glm") & (met.output == "y1"), "holdout_kl"].iloc[0]
        setting = dict(zip(SETTING + ["run_id"],
                           [float(m[1]), int(m[2]), float(m[3]), float(m[4]), int(m[5])]))
        cells.append((setting, int(pd.DataFrame(d["resources"])["n_train"].iloc[0]), float(kl)))
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", default="./sim_waveome_output/cells")
    ap.add_argument("--out", default="./sim_waveome_output/baseline_refit.csv")
    ap.add_argument("--jobs", type=int, default=1, help="cells refit in parallel")
    args = ap.parse_args()

    key = SETTING + ["run_id"]
    done = set()
    if os.path.exists(args.out):
        done = set(map(tuple, pd.read_csv(args.out)[key].drop_duplicates().values.tolist()))
    todo = [c for c in finished_cells(args.cells)
            if tuple(c[0][k] for k in key) not in done]
    print(f"{len(todo)} cells to refit ({len(done)} already in {args.out})")

    n_new = n_bad = 0
    for rows in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(refit_cell)(*c) for c in todo):
        if not len(rows):
            continue
        # One cell per write, so an interrupted run loses at most one cell
        rows.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        n_new += 1
        n_bad += not rows.verified.iloc[0]
        if n_new % 100 == 0:
            print(f"  {n_new}/{len(todo)} cells, {n_bad} failed the data check", flush=True)

    out = pd.read_csv(args.out)
    cells = out.drop_duplicates(key)
    print(f"{args.out}: {len(cells)} cells, {int((~cells.verified).sum())} failed the data check")
    for model, g in out[out.verified].groupby("model"):
        print(f"{model} status:", g.status.value_counts().to_dict())


if __name__ == "__main__":
    main()
