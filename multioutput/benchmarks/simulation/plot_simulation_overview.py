"""
Generate a 4-panel overview figure of the simulation data-generating process.

Panels:
  A  Observed trajectories for 4 active-pathway metabolites under a spike
     effect, showing NB noise and subject variability across 5 subjects.
  B  Same metabolites under a linear trend effect.
  C  Observed trajectories comparing a nuisance (periodic) metabolite against
     a null background metabolite (no signal).
  D  Annotation fraction schematic: pathway members as circles, annotated
     subset filled, withheld subset hollow.

Usage (run from multioutput/benchmarks/):
    python -m simulation.plot_simulation_overview \
        --output multioutput_manuscript/figures/simulation_overview.pdf
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from .core import simulate_longitudinal_data
from .effects import (
    create_spike_effect,
    create_linear_increase_effect,
    create_periodic_effect,
)

N_DISPLAY_SUBJECTS = 5  # subjects shown per panel — enough to show variability without clutter


def _select_responsive_mets(df, candidate_mets, effect_window=(8, 12), n=4):
    """Pick n/2 metabolites with the strongest positive and n/2 with strongest
    negative response in the effect window, illustrating within-pathway heterogeneity."""
    scores = {}
    for m in candidate_mets:
        sub = df[df["metabolite_id"] == m]
        in_window  = sub[sub["time"].between(*effect_window)]["true_mu"].mean()
        out_window = sub[~sub["time"].between(*effect_window)]["true_mu"].mean()
        scores[m] = in_window - out_window
    ranked = sorted(scores, key=scores.get, reverse=True)
    n_pos = n // 2
    n_neg = n - n_pos
    return ranked[:n_pos] + ranked[-n_neg:]


def _select_trend_mets(df, candidate_mets, n=4):
    """Pick n/2 metabolites with the strongest positive and n/2 with the strongest
    negative linear trend, illustrating within-pathway heterogeneity."""
    scores = {}
    for m in candidate_mets:
        sub = df[df["metabolite_id"] == m].sort_values("time")
        if len(sub) < 2:
            scores[m] = 0.0
            continue
        t = sub["time"].values
        mu = sub["true_mu"].values
        scores[m] = np.polyfit(t, mu, 1)[0]  # slope
    ranked = sorted(scores, key=scores.get, reverse=True)
    n_pos = n // 2
    n_neg = n - n_pos
    return ranked[:n_pos] + ranked[-n_neg:]


def _trajectories_for(df, metabolite_ids, subject_ids):
    """Return dict met_id -> list of (times, obs, true_mu) per subject."""
    out = {m: [] for m in metabolite_ids}
    for sid in subject_ids:
        sub = df[df["subject_id"] == sid].sort_values("time")
        for mid in metabolite_ids:
            met = sub[sub["metabolite_id"] == mid]
            if not met.empty:
                out[mid].append((met["time"].values, met["value"].values,
                                 met["true_mu"].values))
    return out


def _plot_trajectories(ax, traj_dict, palette, alpha_obs=0.3, alpha_mu=0.85, lw_mu=2.0):
    for (mid, trajs), color in zip(traj_dict.items(), palette):
        first = True
        for times, obs, mu in trajs:
            ax.scatter(times, np.log1p(obs), color=color, alpha=alpha_obs, s=14, zorder=2)
            ax.plot(times, np.log1p(mu), color=color, alpha=alpha_mu,
                    lw=lw_mu, label=mid if first else None, zorder=3)
            first = False


def make_figure(output_path: str, seed: int = 42):
    rng_state = np.random.get_state()
    np.random.seed(seed)

    N_SUBJECTS = 20
    N_MET = 100
    N_PW = 5
    MPP = 10
    TIME_PTS = np.linspace(0, 20, 8)
    NUISANCE_AMP = 1.5
    NUISANCE_PERIOD = 10.0
    ANNOTATION_FRAC = 0.7
    EFFECT_MAG = 2.5

    nuisance_func = create_periodic_effect(NUISANCE_AMP, NUISANCE_PERIOD)

    # Spike condition
    spike_func = create_spike_effect(8, 12, EFFECT_MAG)
    df_spike, true_pws_spike, _ = simulate_longitudinal_data(
        n_subjects=N_SUBJECTS, n_metabolites=N_MET, n_pathways=N_PW,
        metabolites_per_pathway=MPP, time_points=TIME_PTS,
        effect_func=spike_func, nuisance_fraction=0.2,
        nuisance_effect_func=nuisance_func, annotation_fraction=ANNOTATION_FRAC,
        metabolite_effect_sd=0.5,  # larger scaler SD so individual mets show clear signal
    )

    np.random.seed(seed + 1)
    slope = EFFECT_MAG / (20 - 6)
    linear_func = create_linear_increase_effect(slope, intercept_time=6)
    df_linear, true_pws_linear, _ = simulate_longitudinal_data(
        n_subjects=N_SUBJECTS, n_metabolites=N_MET, n_pathways=N_PW,
        metabolites_per_pathway=MPP, time_points=TIME_PTS,
        effect_func=linear_func, nuisance_fraction=0.2,
        nuisance_effect_func=nuisance_func, annotation_fraction=ANNOTATION_FRAC,
        metabolite_effect_sd=0.5,
    )

    active_pid = list(true_pws_spike.keys())[0]
    active_mets_all = true_pws_spike[active_pid]

    # Pick metabolites that show clear upward response
    display_mets_spike  = _select_responsive_mets(df_spike,  active_mets_all, n=4)
    display_mets_linear = _select_trend_mets(df_linear, true_pws_linear[active_pid], n=4)

    all_subjects = sorted(df_spike["subject_id"].unique())
    display_sids = all_subjects[:N_DISPLAY_SUBJECTS]

    # Background / nuisance metabolites
    all_pathway_mets = set(m for mets in true_pws_spike.values() for m in mets)
    background_mets = [m for m in df_spike["metabolite_id"].unique()
                       if m not in all_pathway_mets]

    # Nuisance: highest true_mu std among background (periodic signal adds variance)
    nuisance_candidate = max(
        background_mets[:40],
        key=lambda m: df_spike[df_spike["metabolite_id"] == m]["true_mu"].std()
    )
    # Null: lowest true_mu std
    null_candidate = min(
        background_mets[40:80],
        key=lambda m: df_spike[df_spike["metabolite_id"] == m]["true_mu"].std()
    )

    # ---- Figure layout ---------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35,
                  height_ratios=[1.0, 1.0])

    ax_spike  = fig.add_subplot(gs[0, 0])
    ax_linear = fig.add_subplot(gs[0, 1])
    ax_noise  = fig.add_subplot(gs[1, 0])
    ax_annot  = fig.add_subplot(gs[1, 1])

    palette_active = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    palette_noise  = ["#ff7f00", "#999999"]

    # ---- Panel A: Spike --------------------------------------------------
    traj_spike = _trajectories_for(df_spike, display_mets_spike, display_sids)
    _plot_trajectories(ax_spike, traj_spike, palette_active)
    ax_spike.axvspan(8, 12, alpha=0.10, color="red", label="Effect window")
    ax_spike.set_title("(A) Active pathway — spike effect", fontsize=11, fontweight="bold")
    ax_spike.set_xlabel("Time")
    ax_spike.set_ylabel("log(abundance + 1)")
    ax_spike.legend(fontsize=7, ncol=2, loc="upper left")

    # ---- Panel B: Linear -------------------------------------------------
    traj_linear = _trajectories_for(df_linear, display_mets_linear,
                                    sorted(df_linear["subject_id"].unique())[:N_DISPLAY_SUBJECTS])
    _plot_trajectories(ax_linear, traj_linear, palette_active)
    ax_linear.set_title("(B) Active pathway — linear trend", fontsize=11, fontweight="bold")
    ax_linear.set_xlabel("Time")
    ax_linear.set_ylabel("log(abundance + 1)")
    ax_linear.legend(fontsize=7, ncol=2, loc="upper left")

    # ---- Panel C: Nuisance vs null ---------------------------------------
    noise_dict = {"Nuisance (periodic)": nuisance_candidate,
                  "Null (background)":   null_candidate}
    traj_noise = _trajectories_for(df_spike, list(noise_dict.values()), display_sids)
    traj_relabeled = {lab: traj_noise[mid] for lab, mid in noise_dict.items()}
    _plot_trajectories(ax_noise, traj_relabeled, palette_noise)
    t_smooth = np.linspace(0, 20, 200)
    # Approximate displayed baseline as mean true_mu at t=0 for nuisance met
    base_mu = df_spike[df_spike["metabolite_id"] == nuisance_candidate]["true_mu"].mean()
    periodic_overlay = np.log1p(base_mu * np.exp(NUISANCE_AMP * np.sin(2 * np.pi * t_smooth / NUISANCE_PERIOD)))
    ax_noise.plot(t_smooth, periodic_overlay, color=palette_noise[0],
                  lw=1.5, linestyle="--", alpha=0.6, label="$h(t)$ (true)")
    ax_noise.set_title("(C) Nuisance vs. null metabolite", fontsize=11, fontweight="bold")
    ax_noise.set_xlabel("Time")
    ax_noise.set_ylabel("log(abundance + 1)")
    ax_noise.legend(fontsize=8, loc="upper right")

    # ---- Panel D: Annotation fraction schematic --------------------------
    ax_annot.set_xlim(-1, 11)
    ax_annot.set_ylim(-3.0, 3.0)
    ax_annot.axis("off")
    ax_annot.set_title(r"(D) Annotation fraction ($\rho = 0.7$)",
                       fontsize=11, fontweight="bold")

    n_members = 10
    n_annotated = int(n_members * ANNOTATION_FRAC)  # 7
    xs = np.linspace(0, 10, n_members)
    y_members = 1.0   # circles sit in the upper half; annotations go below
    radius = 0.42

    # Pathway ellipse + label above
    ellipse = mpatches.Ellipse((5, y_members), width=12.0, height=1.5,
                                facecolor="#f0f0f0", edgecolor="#aaaaaa",
                                linewidth=1.5, zorder=0)
    ax_annot.add_patch(ellipse)
    ax_annot.text(5, y_members + 1.05, "True pathway (all 10 members)",
                  ha="center", va="bottom", fontsize=9, color="#555555")

    for i, x in enumerate(xs):
        annotated = i < n_annotated
        circle = plt.Circle((x, y_members), radius,
                             color="#377eb8" if annotated else "white",
                             ec="#377eb8", lw=1.5, zorder=2)
        ax_annot.add_patch(circle)
        ax_annot.text(x, y_members, f"M{i+1}", ha="center", va="center",
                      fontsize=6, color="white" if annotated else "#377eb8",
                      fontweight="bold", zorder=3)

    # Both annotations go below the ellipse so the panel title is unobstructed.
    # Pathway-aware: bracket spanning annotated members
    mid_annot_x = xs[:n_annotated].mean()
    ax_annot.annotate(
        "", xy=(xs[n_annotated - 1] + radius + 0.05, y_members - 0.9),
        xytext=(xs[0] - radius - 0.05, y_members - 0.9),
        arrowprops=dict(arrowstyle="-", color="#377eb8", lw=1.3),
    )
    ax_annot.plot([xs[0] - radius - 0.05, xs[0] - radius - 0.05],
                  [y_members - 0.75, y_members - 0.9], color="#377eb8", lw=1.3)
    ax_annot.plot([xs[n_annotated - 1] + radius + 0.05,
                   xs[n_annotated - 1] + radius + 0.05],
                  [y_members - 0.75, y_members - 0.9], color="#377eb8", lw=1.3)
    ax_annot.text(mid_annot_x, y_members - 1.0,
                  "Pathway-aware methods see\nonly annotated members (filled)",
                  ha="center", va="top", fontsize=7.5, color="#377eb8")

    # Data-driven: bracket spanning withheld members
    mid_withheld_x = xs[n_annotated:].mean()
    ax_annot.annotate(
        "", xy=(xs[-1] + radius + 0.05, y_members - 0.9),
        xytext=(xs[n_annotated] - radius - 0.05, y_members - 0.9),
        arrowprops=dict(arrowstyle="-", color="#e41a1c", lw=1.3),
    )
    ax_annot.plot([xs[n_annotated] - radius - 0.05,
                   xs[n_annotated] - radius - 0.05],
                  [y_members - 0.75, y_members - 0.9], color="#e41a1c", lw=1.3)
    ax_annot.plot([xs[-1] + radius + 0.05, xs[-1] + radius + 0.05],
                  [y_members - 0.75, y_members - 0.9], color="#e41a1c", lw=1.3)
    ax_annot.text(mid_withheld_x, y_members - 1.0,
                  "Data-driven methods can\ndiscover withheld members (hollow)",
                  ha="center", va="top", fontsize=7.5, color="#e41a1c")

    # ---- Save ------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")

    np.random.set_state(rng_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="multioutput_manuscript/figures/simulation_overview.pdf",
        help="Output file path (.pdf or .png)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_figure(args.output, seed=args.seed)
