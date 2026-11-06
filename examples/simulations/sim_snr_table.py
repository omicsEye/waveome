"""Signal share of each (epsilon, alpha) noise setting in the simulation grid.

Signal share is the squared correlation between the true latent signal f and
log(y + 1), the fraction of an outcome's log-count variation the signal
explains. It combines both noise sources: latent noise (sd epsilon, added to f
on the log scale) and negative binomial overdispersion (alpha). It is computed
by simulation, from the same sim_data the sweep uses, so it reflects the zeros
and the log(y + 1) transform as the methods see them.

Writes one row per simulated outcome (--out) and a summary per setting
(--summary): median signal share pooled over outcomes, per-outcome medians,
the zero fraction and the largest count, plus two Monte Carlo checks on the
pooled median -- a bootstrap standard error over seeds, and the median from
the first half of the seeds against all of them.

Usage, from examples/simulations:
    python sim_snr_table.py --seeds 1000 --jobs 8
    python sim_snr_table.py --seeds 100 --sizes 500x4 --jobs 8 \
        --out ./sim_waveome_output/snr_table_rows_500x4.csv \
        --summary ./sim_waveome_output/snr_table_500x4.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import sim_waveome_hpc_run as h

OUTCOMES = ["y1", "y2", "y3", "y4"]


def one_seed(seed, sizes, eps_list, alpha_list):
    rows = []
    for units, rate in sizes:
        for eps in eps_list:
            for alpha in alpha_list:
                df, params = h.sim_data(kern_out=h.kern_out, random_seed=seed,
                                        num_units=units, rate=rate, eps=eps,
                                        alpha=alpha)
                if df is None:
                    continue
                for k in OUTCOMES:
                    y = df[k].to_numpy(dtype=float)
                    rows.append({
                        "units": units, "rate": rate, "epsilon": eps,
                        "alpha": alpha, "seed": seed, "outcome": k,
                        "n_obs": len(y),
                        "signal_share": np.corrcoef(params[k]["f"], np.log(y + 1))[0, 1] ** 2,
                        "zero_fraction": np.mean(y == 0),
                        "log10_max_count": np.log10(y.max() + 1),
                    })
    return rows


def summarize(d, n_boot=2000, rng_seed=9102):
    rng = np.random.default_rng(rng_seed)
    out = []
    for (eps, alpha), g in d.groupby(["epsilon", "alpha"]):
        # One value per seed (pooled over sizes and outcomes) for the bootstrap
        seeds = g.seed.unique()
        by_seed = {s: v.to_numpy() for s, v in g.groupby("seed").signal_share}
        # nanmedian: an outcome that is all zeros has no defined correlation
        boot = [np.nanmedian(np.concatenate([by_seed[s] for s in rng.choice(seeds, len(seeds))]))
                for _ in range(n_boot)]
        half = g[g.seed.isin(seeds[: len(seeds) // 2])]
        per_out = g.groupby("outcome").signal_share.median()
        out.append({
            "epsilon": eps, "alpha": alpha, "n_seeds": len(seeds),
            "n_all_zero_outcomes": int(g.signal_share.isna().sum()),
            "signal_share_median": g.signal_share.median(),
            "signal_share_q25": g.signal_share.quantile(0.25),
            "signal_share_q75": g.signal_share.quantile(0.75),
            "mc_se_median": np.std(boot),
            "median_first_half_seeds": half.signal_share.median(),
            **{f"median_{k}": per_out[k] for k in OUTCOMES},
            "zero_fraction_median": g.zero_fraction.median(),
            "log10_max_count_median": g.log10_max_count.median(),
            **{f"median_{u}x{r}": v for (u, r), v in
               g.groupby(["units", "rate"]).signal_share.median().items()},
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    grid = h.make_grid([0])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=1000)
    # Signal share does not depend on size (each outcome's per-observation
    # noise is fixed by epsilon and alpha); 500x4 is ~15x the cost of the
    # rest combined, so it is checked with fewer seeds via --sizes
    ap.add_argument("--sizes", default="10x4,50x8,100x8",
                    help="units x rate pairs, comma-separated")
    ap.add_argument("--epsilon", type=float, nargs="+",
                    default=sorted(grid.epsilon.unique()))
    ap.add_argument("--alpha", type=float, nargs="+",
                    default=sorted(grid.alpha.unique()))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="./sim_waveome_output/snr_table_rows.csv")
    ap.add_argument("--summary", default="./sim_waveome_output/snr_table.csv")
    ap.add_argument("--from-rows", action="store_true",
                    help="summarize the rows already in --out; no simulation")
    args = ap.parse_args()

    if args.from_rows:
        d = pd.read_csv(args.out)
    else:
        sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
        rows = Parallel(n_jobs=args.jobs)(
            delayed(one_seed)(s, sizes, args.epsilon, args.alpha) for s in range(args.seeds))
        d = pd.DataFrame([r for chunk in rows for r in chunk])
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        d.to_csv(args.out, index=False)
    s = summarize(d)
    s.to_csv(args.summary, index=False)
    pd.set_option("display.width", 250)
    print(s.round(3).to_string(index=False))
