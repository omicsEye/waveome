"""Explain scale-stratified pooling visually, using the real M=400 draws.

Four panels:
  A  every metabolite has its OWN permutation null, and their scales differ
     by orders of magnitude (SD 0 to ~21 in this run)
  B  naive pooling builds ONE tail from all of them -- dominated by the
     point-mass majority, so a wide-null metabolite is judged against a
     reference far tighter than its own
  C  stratified pooling builds one tail PER SCALE BIN
  D  the consequence, measured: false-positive rate by subgroup

Everything is null by construction in this data, so every rate in panel D
should sit at the nominal 0.05 line.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DRAWS = "sim_waveome_output/t2_m400strat_raw_draws.csv"
TIE_TOL, DEGEN_TOL, WIDE_TOL = 1e-3, 1e-6, 0.5

# validated palette (dataviz validate_palette.js, light mode: all pass)
C_POOL = "#2563eb"    # pooled / naive
C_STRAT = "#d97706"   # stratified
C_OBS = "#64748b"     # neutral ink for observed data
C_THIRD = "#059669"


def load(kt="lin"):
    d = pd.read_csv(DRAWS)
    g = d[d.kernel_type == kt]
    obs = g[g.draw == -1].set_index("outcome")["log_bf"]
    nul = g[g.draw >= 0]
    centre = nul.groupby("outcome")["log_bf"].median()
    spread = nul.groupby("outcome")["log_bf"].std()
    by = {o: v["log_bf"].values for o, v in nul.groupby("outcome")}
    common = [o for o in obs.index if o in by]
    return obs[common], centre[common], spread[common], {o: by[o] for o in common}


def binof(s):
    return 0 if s < DEGEN_TOL else (1 if s < WIDE_TOL else 2)


def rates(obs, centre, spread, by, mode):
    """false-positive rate by subgroup under 'pooled' or 'stratified'."""
    bins = {o: binof(spread[o]) for o in obs.index}
    if mode == "pooled":
        pool = np.concatenate([np.asarray(v) - centre[o] for o, v in by.items()])
        pools = {0: pool, 1: pool, 2: pool}
    else:
        acc = {0: [], 1: [], 2: []}
        for o, v in by.items():
            acc[bins[o]].extend(np.asarray(v) - centre[o])
        pools = {k: np.asarray(v) for k, v in acc.items()}
    p = {}
    for o in obs.index:
        pl = pools[bins[o]]
        e = obs[o] - centre[o]
        p[o] = (1 + np.sum(pl >= e - TIE_TOL)) / (1 + len(pl))
    p = pd.Series(p)
    out = []
    for b in (0, 1, 2):
        sel = [o for o in obs.index if bins[o] == b]
        out.append(np.mean(p[sel] < 0.05) if sel else np.nan)
    return out


def main():
    obs, centre, spread, by = load("lin")
    bins = pd.Series({o: binof(spread[o]) for o in obs.index})
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))

    # ---- A: individual nulls, sorted by scale ----
    a = ax[0, 0]
    order = spread.sort_values().index
    picks = [order[int(f * (len(order) - 1))] for f in
             (0.05, 0.30, 0.55, 0.68, 0.80, 0.88, 0.94, 0.98, 1.0)]
    for i, o in enumerate(picks):
        v = np.asarray(by[o]) - centre[o]
        col = [C_OBS, C_THIRD, C_STRAT][binof(spread[o])]
        a.scatter(v, np.full(len(v), i), s=14, alpha=0.55, color=col)
        a.text(-24, i, f"SD={spread[o]:.2f}", fontsize=7, va="center", ha="left",
               color=col)
    a.set_yticks([])
    a.set_xlim(-25, 12)
    a.set_xlabel("null draw, centred on its own median", fontsize=9)
    a.set_title("A. Each metabolite has its own null.\n"
                "Scales differ by orders of magnitude (SD 0 → 21)", fontsize=10)
    a.axvline(0, color="#94a3b8", lw=1)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)

    # ---- B: naive pooling ----
    b = ax[0, 1]
    pool = np.concatenate([np.asarray(v) - centre[o] for o, v in by.items()])
    b.hist(pool, bins=np.linspace(-25, 12, 90), color=C_POOL, alpha=0.65,
           label=f"one pooled tail (n={len(pool):,})")
    wide_o = spread.idxmax()
    own = np.asarray(by[wide_o]) - centre[wide_o]
    b.hist(own, bins=np.linspace(-25, 12, 90), color=C_STRAT, alpha=0.85,
           label=f"one WIDE metabolite's own null (SD={spread[wide_o]:.1f})")
    b.set_yscale("log")
    b.set_xlabel("centred log_bf", fontsize=9)
    b.set_ylabel("count (log)", fontsize=9)
    b.set_title("B. Naive pooling: ONE tail for everyone.\n"
                "63% of draws are point masses at 0, so the pooled tail is far\n"
                "tighter than a wide metabolite's own null", fontsize=10)
    b.legend(fontsize=7.5, frameon=False)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)

    # ---- C: stratified pools ----
    c = ax[1, 0]
    labs = ["degenerate (SD<1e-6)", "narrow (1e-6–0.5)", "WIDE (SD≥0.5)"]
    cols = [C_OBS, C_THIRD, C_STRAT]
    styles = ["-", "--", ":"]
    for bi in (0, 1, 2):
        sel = [o for o in obs.index if bins[o] == bi]
        pl = np.concatenate([np.asarray(by[o]) - centre[o] for o in sel])
        hist, edges = np.histogram(pl, bins=np.linspace(-25, 12, 90))
        ctr = 0.5 * (edges[1:] + edges[:-1])
        c.plot(ctr, np.maximum(hist, 0.5), color=cols[bi], ls=styles[bi], lw=2,
               label=f"{labs[bi]}  n_out={len(sel)}, n_draws={len(pl):,}")
    c.set_yscale("log")
    c.set_xlabel("centred log_bf", fontsize=9)
    c.set_ylabel("count (log)", fontsize=9)
    c.set_title("C. Stratified: one tail PER SCALE BIN.\n"
                "A metabolite is only compared against others of similar scale",
                fontsize=10)
    c.legend(fontsize=7.5, frameon=False)
    for s in ("top", "right"):
        c.spines[s].set_visible(False)

    # ---- D: measured consequence ----
    d = ax[1, 1]
    r_pool = rates(obs, centre, spread, by, "pooled")
    r_str = rates(obs, centre, spread, by, "stratified")
    x = np.arange(3)
    d.bar(x - 0.19, r_pool, 0.36, color=C_POOL, label="naive pooling")
    d.bar(x + 0.19, r_str, 0.36, color=C_STRAT, label="scale-stratified")
    d.axhline(0.05, color="#334155", lw=1.5, ls="--")
    d.text(2.42, 0.055, "nominal 0.05", fontsize=8, ha="right", color="#334155")
    for xi, (p_, s_) in enumerate(zip(r_pool, r_str)):
        if not np.isnan(p_):
            d.text(xi - 0.19, p_ + 0.008, f"{p_:.3f}", ha="center", fontsize=8)
        if not np.isnan(s_):
            d.text(xi + 0.19, s_ + 0.008, f"{s_:.3f}", ha="center", fontsize=8)
    d.set_xticks(x)
    d.set_xticklabels([f"degenerate\nn={int((bins==0).sum())}",
                       f"narrow\nn={int((bins==1).sum())}",
                       f"WIDE\nn={int((bins==2).sum())}"], fontsize=8.5)
    d.set_ylabel("false-positive rate at p<0.05", fontsize=9)
    d.set_title("D. Consequence, measured on this data.\n"
                "All outcomes are null, so every bar should sit on the line",
                fontsize=10)
    d.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        d.spines[s].set_visible(False)
    d.grid(alpha=0.15, axis="y", lw=0.5)

    fig.suptitle("Scale-stratified pooling: why the permutation nulls get binned "
                 "before p-values are computed", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("sim_waveome_output/stratified_pooling_explained.png", dpi=135)
    print("wrote sim_waveome_output/stratified_pooling_explained.png")
    print(f"pooled     rates: {[round(v,3) for v in r_pool]}")
    print(f"stratified rates: {[round(v,3) for v in r_str]}")


if __name__ == "__main__":
    main()
