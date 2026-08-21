"""Score three ways of turning permutation draws into p-values.

  (a)  scale-stratified pooling            -- current method
  (a-loo) same, but a metabolite's own draws are EXCLUDED from the pool it
          is judged against. IHW (Ignatiadis & Huber 2016) splits into folds
          precisely so a hypothesis's weight never depends on its own
          p-value; our pool has the same circularity and this removes it.
  (c)  conditional QUANTILE REGRESSION on scale -- no bins at all. Pool every
       centred draw tagged with its metabolite's null SD, fit Q_tau(draw | SD)
       over a grid of tau, and read off the tau at which the fitted
       conditional quantile reaches the observed excess. p = 1 - tau.

Run on both existing datasets -- pure arithmetic, no refits:
  * Stage 1 M=400 (everything null)  -> false-positive rate by subgroup,
    which should sit at 0.05 everywhere
  * Stage 2 (112 known true positives) -> realized FDR vs nominal, and power
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from waveome.utilities import calc_bh_qvalues

S1 = "sim_waveome_output/t2_m400strat_raw_draws.csv"
S2 = "sim_waveome_output/t2_stage2_raw_draws.csv"
S2_TRUTH = "sim_waveome_output/t2_stage2_truth.csv"
TIE_TOL, DEGEN_TOL, WIDE_TOL = 1e-3, 1e-6, 0.5
TAUS = 1 - np.concatenate([np.linspace(0.5, 0.02, 40),
                           np.logspace(np.log10(0.02), np.log10(2e-4), 40)])


def load(path, kt):
    d = pd.read_csv(path)
    g = d[d.kernel_type == kt]
    obs = g[g.draw == -1].set_index("outcome")["log_bf"]
    nul = g[g.draw >= 0]
    centre = nul.groupby("outcome")["log_bf"].median()
    spread = nul.groupby("outcome")["log_bf"].std()
    by = {o: v["log_bf"].values for o, v in nul.groupby("outcome")}
    common = [o for o in obs.index if o in by]
    return (obs[common], centre[common], spread[common].fillna(0.0),
            {o: by[o] for o in common})


def binof(s):
    return 0 if s < DEGEN_TOL else (1 if s < WIDE_TOL else 2)


def p_stratified(obs, centre, spread, by, loo=False):
    bins = {o: binof(spread[o]) for o in obs.index}
    pools = {0: [], 1: [], 2: []}
    for o, v in by.items():
        pools[bins[o]].extend(np.asarray(v) - centre[o])
    pools = {k: np.asarray(v) for k, v in pools.items()}
    out = {}
    for o in obs.index:
        e = obs[o] - centre[o]
        pool = pools[bins[o]]
        n_ge = int(np.sum(pool >= e - TIE_TOL))
        n = len(pool)
        if loo:
            own = np.asarray(by[o]) - centre[o]
            n_ge -= int(np.sum(own >= e - TIE_TOL))
            n -= len(own)
        out[o] = (1 + n_ge) / (1 + max(n, 1))
    return pd.Series(out)


def p_quantreg(obs, centre, spread, by):
    """Fit Q_tau(centred draw | null SD) and invert for each observation."""
    ys, xs = [], []
    for o, v in by.items():
        ys.append(np.asarray(v) - centre[o])
        xs.append(np.full(len(v), spread[o]))
    y = np.concatenate(ys)
    X = sm.add_constant(np.concatenate(xs))
    mod = sm.QuantReg(y, X)
    coefs = []
    for t in TAUS:
        try:
            coefs.append(mod.fit(q=t, max_iter=2000).params)
        except Exception:
            coefs.append(coefs[-1] if coefs else np.array([0.0, 0.0]))
    coefs = np.asarray(coefs)                      # (n_tau, 2)
    out = {}
    for o in obs.index:
        e = obs[o] - centre[o]
        q = coefs[:, 0] + coefs[:, 1] * spread[o]   # fitted quantile per tau
        q = np.maximum.accumulate(q)                # enforce monotone in tau
        idx = np.searchsorted(q, e - TIE_TOL)
        if idx >= len(TAUS):
            out[o] = 1.0 - TAUS[-1]                 # beyond the grid
        else:
            out[o] = max(1.0 - TAUS[idx], 1e-6)
    return pd.Series(out)


def report_null(name, p, spread):
    print(f"    {name:22s}", end="")
    for lab, m in [("degen", spread < DEGEN_TOL),
                   ("narrow", (spread >= DEGEN_TOL) & (spread < WIDE_TOL)),
                   ("WIDE", spread >= WIDE_TOL)]:
        s = p[[o for o in p.index if m[o]]]
        print(f"  {lab}={np.mean(s < 0.05):.3f}" if len(s) else f"  {lab}=  n/a",
              end="")
    print(f"   overall={np.mean(p < 0.05):.3f}")


def report_fdr(name, p, tp):
    row = f"    {name:22s}"
    for q in (0.01, 0.05, 0.10):
        qv = calc_bh_qvalues(p.values)
        sel = qv < q
        n = int(sel.sum())
        f = int((sel & ~tp.values).sum())
        row += f"  q={q:<4} FDR={f/n if n else 0:.3f} pow={int((sel & tp.values).sum())/max(int(tp.sum()),1):.3f}"
    print(row)


def main():
    print("=" * 96)
    print("STAGE 1 (M=400, everything null) -- false-positive rate at p<0.05, "
          "nominal 0.05 everywhere")
    print("=" * 96)
    for kt in ("lin", "squared_exponential"):
        obs, centre, spread, by = load(S1, kt)
        print(f"  {kt}:")
        report_null("(a) stratified", p_stratified(obs, centre, spread, by), spread)
        report_null("(a-loo) leave-one-out",
                    p_stratified(obs, centre, spread, by, loo=True), spread)
        report_null("(c) quantile reg", p_quantreg(obs, centre, spread, by), spread)

    print()
    print("=" * 96)
    print("STAGE 2 (112 true positives) -- realized FDR vs nominal, and power")
    print("=" * 96)
    truth = pd.read_csv(S2_TRUTH, index_col=0)
    for kt in ("lin", "squared_exponential"):
        obs, centre, spread, by = load(S2, kt)
        tp = truth.loc[obs.index, "is_tp"].astype(bool)
        print(f"  {kt}:")
        report_fdr("(a) stratified", p_stratified(obs, centre, spread, by), tp)
        report_fdr("(a-loo) leave-one-out",
                   p_stratified(obs, centre, spread, by, loo=True), tp)
        report_fdr("(c) quantile reg", p_quantreg(obs, centre, spread, by), tp)


if __name__ == "__main__":
    main()
