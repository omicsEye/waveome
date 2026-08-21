"""Supplemental figure: the permutation-null significance method on iHMP.

Publication build -- 9 pt type matching the manuscript rcParams, two-column
width, bold panel letters, terse axis labels (the caption carries the
explanation), vector PDF alongside a high-DPI PNG.

Results are merged across the two runs: hbi at B=60, time_from_max at B=120.
That stratum was re-run at higher resolution because its p-floor cleared
Benjamini-Hochberg's rank-1 threshold by only 3%; the counts were unchanged,
so the figure reports the better-resolved version.

Supersedes the simulation figure (stratified_pooling_explained.png), which
illustrated scale-BINNED pooling -- tested, then replaced by conditional
quantile regression, so it no longer depicts the pipeline.

    python plot_permutation_null_ihmp.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

DRAWS = "output/ihmp_permutation_draws.csv"
RESULTS = "output/ihmp_permutation_significance.csv"
RESULTS_TFM = "output/ihmp_permutation_tfm_b120.csv"
OUT = "output/permutation_null_explained"
FOCUS = ("hbi", "lin")          # stratum shown in panels A-C
Q_TARGET = 0.10
M_TESTS = 564

# validated categorical pair (dataviz validate_palette.js, light mode: pass)
C_NULL = "#2563eb"
C_SIG = "#d97706"
C_OBS = "#64748b"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    # embed real fonts rather than Type 3, which many journals reject
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def panel_letter(ax, letter):
    ax.text(-0.15, 1.07, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")


def load_results():
    """hbi from the B=60 run, time_from_max from its B=120 re-run."""
    base = pd.read_csv(RESULTS)
    tfm = pd.read_csv(RESULTS_TFM)
    return pd.concat([base[base.covariate != "time_from_max"], tfm],
                     ignore_index=True)


def main():
    d = pd.read_csv(DRAWS)
    res = load_results()

    g = d[(d.covariate == FOCUS[0]) & (d.kernel_type == FOCUS[1])]
    obs = g[g.draw == -1].set_index("metabolite")["log_bf"]
    nul = g[g.draw >= 0]
    centre = nul.groupby("metabolite")["log_bf"].median()
    spread = nul.groupby("metabolite")["log_bf"].std().fillna(0.0)
    common = [m for m in obs.index if m in centre.index]
    obs, centre, spread = obs[common], centre[common], spread[common]
    rf = res[(res.covariate == FOCUS[0]) & (res.kernel_type == FOCUS[1])]
    sig = (rf.set_index("metabolite").reindex(common).q_value
           < Q_TARGET).to_numpy()

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.8))

    # A -- per-metabolite nulls across the scale range
    a = ax[0, 0]
    order = spread.sort_values().index
    picks = [order[int(f * (len(order) - 1))] for f in
             (0.10, 0.45, 0.62, 0.74, 0.83, 0.90, 0.95, 0.985, 1.0)]
    for i, m in enumerate(picks):
        v = nul[nul.metabolite == m]["log_bf"].to_numpy() - centre[m]
        a.scatter(v, np.full(len(v), i), s=5, alpha=0.45, color=C_OBS,
                  linewidths=0)
    a.axvline(0, color="#94a3b8", lw=0.8)
    # Row identifier belongs on the left as a tick label, matching the other
    # three panels; as right-hand annotation it also forced the x-range wide
    # enough to squeeze the data into part of the panel.
    a.set_xlim(-7.5, 7.5)
    a.set_ylim(-0.8, len(picks) - 0.2)
    a.set_yticks(range(len(picks)))
    a.set_yticklabels([f"{spread[m]:.2f}" for m in picks], fontsize=6.5)
    a.set_ylabel("null SD of that metabolite")
    a.set_xticks([-5, 0, 5])
    a.set_xlabel("centred null draw (log BF)")
    panel_letter(a, "A")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

    # B -- the fitted conditional quantiles
    b = ax[0, 1]
    y = np.concatenate([nul[nul.metabolite == m]["log_bf"].to_numpy()
                        - centre[m] for m in common])
    x = np.concatenate([np.full((nul.metabolite == m).sum(), spread[m])
                        for m in common])
    b.scatter(x, y, s=1.5, alpha=0.08, color=C_OBS, linewidths=0)
    mod = sm.QuantReg(y, sm.add_constant(x))
    xs = np.linspace(0, x.max(), 100)
    for tau, ls in [(0.50, ":"), (0.95, "--"), (0.99, "-")]:
        p = mod.fit(q=tau, max_iter=2000).params
        b.plot(xs, p[0] + p[1] * xs, color=C_NULL, ls=ls, lw=1.4,
               label=rf"$Q_{{{tau:.2f}}}$")
    b.set_xlabel("null SD of that metabolite")
    b.set_ylabel("centred null draw")
    b.set_ylim(np.percentile(y, 0.2), np.percentile(y, 99.8))
    b.legend(frameon=False, loc="upper left", handlelength=1.6,
             borderpad=0.2, labelspacing=0.25)
    panel_letter(b, "B")
    for s in ("top", "right"):
        b.spines[s].set_visible(False)

    # C -- observed values against the fitted bar
    c = ax[1, 0]
    excess = (obs - centre).to_numpy()
    sd = spread.to_numpy()
    p95 = mod.fit(q=0.95, max_iter=2000).params
    c.scatter(sd[~sig], excess[~sig], s=6, alpha=0.45, color=C_OBS,
              linewidths=0, label=f"n.s. ({int((~sig).sum())})")
    c.scatter(sd[sig], excess[sig], s=11, color=C_SIG, zorder=3,
              linewidths=0, label=f"$q<{Q_TARGET}$ ({int(sig.sum())})")
    c.plot(xs, p95[0] + p95[1] * xs, color=C_NULL, ls="--", lw=1.4,
           label=r"fitted $Q_{0.95}$")
    c.set_yscale("symlog", linthresh=1)
    c.set_xlabel("null SD of that metabolite")
    c.set_ylabel("observed excess (log BF)")
    c.legend(frameon=False, loc="lower right", handlelength=1.6,
             borderpad=0.2, labelspacing=0.25)
    panel_letter(c, "C")
    for s in ("top", "right"):
        c.spines[s].set_visible(False)

    # D -- results, and whether each zero is a limit or a finding
    dd = ax[1, 1]
    npool = {f"{kt}:{cov}": len(gg[gg.draw >= 0])
             for (cov, kt), gg in d.groupby(["covariate", "kernel_type"])}
    rows = []
    for s, gg in res.groupby("stratum"):
        fl = gg.p_value.min()
        rows.append((s.replace("squared_exponential", "SE"),
                     int((gg.q_value < Q_TARGET).sum()), fl,
                     fl <= 1.5 / max(npool.get(s, 2), 2)))
    rows.sort(key=lambda r: r[1])
    ypos = np.arange(len(rows))
    dd.barh(ypos, [r[1] for r in rows], color=C_SIG, height=0.5)
    for i, (_, n, fl, af) in enumerate(rows):
        dd.text(n + 4, i, f"{n}{'*' if af else '†'}", va="center",
                fontsize=7.5, color="#334155")
    dd.set_yticks(ypos)
    dd.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    dd.set_xlim(0, 190)
    dd.set_xlabel(f"metabolites with $q<{Q_TARGET}$ (of {M_TESTS})")
    dd.text(0.98, 0.06,
            "* smallest $p$ at the attainable floor\n"
            "† smallest $p$ above it — not extreme",
            transform=dd.transAxes, fontsize=6.5, ha="right", va="bottom",
            color="#334155")
    panel_letter(dd, "D")
    for s in ("top", "right"):
        dd.spines[s].set_visible(False)
    dd.grid(alpha=0.15, axis="x", lw=0.5)

    # Draft caption, generated from the same arrays the panels are drawn
    # from. Writing it by hand invites drift: an earlier hand-drafted version
    # still quoted time_from_max's pre-top-up p-floor. Numbers here cannot go
    # stale without the figure going stale too.
    b_used = {c: int(gg[gg.draw >= 0]["draw"].max()) + 1
              for c, gg in d.groupby("covariate")}
    at_floor = [r[0] for r in rows if r[3]]
    above = [r[0] for r in rows if not r[3]]
    n_sig_focus = int(sig.sum())
    caption = f"""Supplementary Figure S{{n}}. Permutation-null significance testing.

(A) Permutation nulls for {len(picks)} metabolites in the {FOCUS[1]}:{FOCUS[0]} stratum, each centred on its own median and ordered by
null scale (right-hand column); scales span zero -- components that collapse
under every permutation -- to {spread.max():.2f}.

(B) All {M_TESTS} metabolites' null draws plotted against their own null
standard deviation, with conditional quantiles fitted by quantile regression.
Q0.50 is flat at zero, confirming per-metabolite centring, while the upper
quantiles rise with null width, so the threshold a metabolite must exceed
adapts to its own null rather than to a single pooled distribution.

(C) Observed excess over each metabolite's null centre against that null's
scale; points above the fitted Q0.95 are candidates, with those passing
Benjamini-Hochberg at q < {Q_TARGET} in orange (n = {n_sig_focus}). A metabolite
with a narrow null clears the threshold at an excess near 1 log BF while one
with a wide null requires roughly 5.

(D) Discoveries per (kernel, covariate) stratum at q < {Q_TARGET}. Asterisks mark
strata whose smallest p-value sits at the floor attainable from the pooled
draws ({', '.join(at_floor)}), where additional permutations could alter the
result; the dagger marks a stratum whose smallest p-value lies well above that
floor ({', '.join(above) or 'none'}), indicating an absence of extreme
statistics rather than insufficient resolution.

Permutations per component: {'; '.join(f'{c} B={b}' for c, b in sorted(b_used.items()))}.
Total draws {len(d[d.draw >= 0]):,} across {M_TESTS} metabolites.

--
Generated by plot_permutation_null_ihmp.py -- regenerate rather than edit in
place, or the numbers will drift from the figure. Check before submission:
panel-letter convention, and single- vs double-column width (this is built at
{7.2:.1f} in, i.e. double).
"""
    with open(f"{OUT}_caption.txt", "w") as fh:
        fh.write(caption)

    fig.tight_layout(w_pad=2.4, h_pad=2.2)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=400, bbox_inches="tight")
    print(f"wrote {OUT}.pdf, {OUT}.png and {OUT}_caption.txt")
    for r in rows:
        print(f"  {r[0]:34s} q<{Q_TARGET}: {r[1]:3d}   min p {r[2]:.2e}"
              f"   {'at floor' if r[3] else 'above floor'}")


if __name__ == "__main__":
    main()
