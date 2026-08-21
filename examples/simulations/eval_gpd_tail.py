"""Evaluate GPD tail approximation as a replacement for cross-metabolite pooling.

Knijnenburg, Wessels, Reinders & Shmulevich (2009), "Fewer permutations, more
accurate P-values" (Bioinformatics): fit a generalized Pareto distribution to
the exceedances above a threshold in a SINGLE test's own permutation
distribution, then evaluate it at the observed statistic. That gives p-values
below the 1/(B+1) floor without borrowing draws from other tests -- which
would remove the scale-binning and its arbitrary 0.5 cutpoint entirely.

Their recipe uses the 250 most extreme permutation values as exceedances and
steps down by 10 whenever an Anderson-Darling test rejects the GPD fit. We
only have B=100 draws per outcome here, so this starts far lower and steps
down more finely; part of what the evaluation answers is how few exceedances
still fit, which sets the B a real run would need.

Runs on the existing M=400 draws -- pure arithmetic, no refits. Every outcome
is null by construction (no within-subject cindex effect), so frac(p<0.05)
should be ~0.05 in every subgroup.
"""
import numpy as np
import pandas as pd
from scipy.stats import genpareto, goodness_of_fit

DRAWS = "sim_waveome_output/t2_m400strat_raw_draws.csv"
TIE_TOL = 1e-3
DEGEN_TOL = 1e-6
WIDE_TOL = 0.5
# Switch to GPD only when the empirical tail is too thin to resolve; with >=
# this many exceedances the plain empirical p-value is already trustworthy.
EMPIRICAL_MIN_EXC = 10
AD_ALPHA = 0.05


def gpd_pvalue(null, obs, n_exc_start=40, n_exc_min=8, step=4):
    """Knijnenburg-style GPD tail p-value.

    Returns (p, n_exc, ad_p, method). Falls back to the empirical p-value
    when the tail is thick enough to resolve, or when no exceedance count
    yields an acceptable GPD fit.
    """
    null = np.asarray(null, dtype=float)
    B = len(null)
    n_over = int(np.sum(null >= obs - TIE_TOL))
    if n_over >= EMPIRICAL_MIN_EXC:
        return (1 + n_over) / (1 + B), np.nan, np.nan, "empirical"

    srt = np.sort(null)
    for n_exc in range(min(n_exc_start, B // 2), n_exc_min - 1, -step):
        u = srt[-n_exc - 1]                 # threshold
        exc = srt[-n_exc:] - u              # exceedances above it
        if np.ptp(exc) < TIE_TOL:           # degenerate tail, nothing to fit
            continue
        try:
            res = goodness_of_fit(
                genpareto, exc, known_params={"loc": 0},
                statistic="ad", n_mc_samples=199,
                random_state=np.random.default_rng(9102))
        except Exception:
            continue
        if res.pvalue > AD_ALPHA:
            c, loc, scale = genpareto.fit(exc, floc=0)
            tail = genpareto.sf(obs - u, c, loc=0, scale=scale)
            return (n_exc / B) * tail, n_exc, res.pvalue, "gpd"
    return (1 + n_over) / (1 + B), np.nan, np.nan, "empirical-fallback"


def main():
    d = pd.read_csv(DRAWS)
    for kt, g in d.groupby("kernel_type"):
        obs = g[g.draw == -1].set_index("outcome")["log_bf"]
        nul = g[g.draw >= 0]
        spread = nul.groupby("outcome")["log_bf"].std()
        by_out = {o: v["log_bf"].values for o, v in nul.groupby("outcome")}
        common = [o for o in obs.index if o in by_out]

        rows = []
        for o in common:
            sd = spread[o]
            if sd < DEGEN_TOL:
                # point-mass null: no tail exists to fit, and excess is ~0
                e = obs[o] - np.median(by_out[o])
                p = 1.0 if e <= TIE_TOL else 1.0 / (1 + len(by_out[o]))
                rows.append((o, sd, p, np.nan, np.nan, "degenerate"))
                continue
            p, n_exc, ad, meth = gpd_pvalue(by_out[o], obs[o])
            rows.append((o, sd, p, n_exc, ad, meth))
        r = pd.DataFrame(rows, columns=["outcome", "sd", "p", "n_exc",
                                        "ad_p", "method"])

        print(f"\n=== {kt}:cindex  (all null by construction) ===")
        nz = r[r.method != "degenerate"]
        print(f"  non-degenerate outcomes: {len(nz)}/{len(r)}")
        print("  method used: " + ", ".join(
            f"{k}={v}" for k, v in r.method.value_counts().items()))
        got = nz[nz.method == "gpd"]
        if len(got):
            print(f"  GPD fits accepted: {len(got)}/{len(nz)} "
                  f"({100*len(got)/len(nz):.0f}%)   "
                  f"exceedances used: median={got.n_exc.median():.0f} "
                  f"range=[{got.n_exc.min():.0f}, {got.n_exc.max():.0f}]")
            print(f"  AD p-values of accepted fits: "
                  f"median={got.ad_p.median():.2f}")
            print(f"  smallest GPD p-value: {got.p.min():.2e} "
                  f"(empirical floor would be {1/101:.2e})")
        else:
            print("  GPD fits accepted: 0 -- tails too spiky to fit")
        for lab, m in [("degenerate", r.sd < DEGEN_TOL),
                       ("narrow", (r.sd >= DEGEN_TOL) & (r.sd < WIDE_TOL)),
                       ("WIDE", r.sd >= WIDE_TOL)]:
            s = r[m]
            if not len(s):
                continue
            print(f"    {lab:11s} n={len(s):3d}  frac(p<0.05)="
                  f"{np.mean(s.p < 0.05):.3f}   <- nominal 0.05")
    print("\nbinning comparison (M=400): lin WIDE 0.032, SE WIDE 0.028")


if __name__ == "__main__":
    main()
