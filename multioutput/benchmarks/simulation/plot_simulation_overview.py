"""
Generate a 6-panel (2x3) overview figure of the simulation data-generating process.

Panels:
  A  Observed trajectories for active-pathway metabolites under a spike effect.
  B  Same metabolites under a sustained ramp (trapezoid) effect.
  C  Same metabolites under an acute perturbation (exponential relaxation).
  D  Nuisance (periodic) metabolite vs. null background metabolite.
  E  Group covariate: one metabolite showing full effect (group 0) vs.
     50% attenuated effect (group 1).
  F  Annotation fraction schematic.

Usage (from multioutput/benchmarks/):
    python -m simulation.plot_simulation_overview \
        --output figures/simulation_overview.pdf
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from .core import simulate_longitudinal_data
from .effects import (
    create_spike_effect,
    create_trapezoid_effect,
    create_perturbation_effect,
    create_periodic_effect,
)

N_DISPLAY_SUBJECTS = 3  # subjects shown per panel — enough to show variability without clutter


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
    import os
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

    # Ramp (trapezoid) condition
    np.random.seed(seed + 1)
    ramp_func = create_trapezoid_effect(5, 13, EFFECT_MAG)
    df_ramp, true_pws_ramp, _ = simulate_longitudinal_data(
        n_subjects=N_SUBJECTS, n_metabolites=N_MET, n_pathways=N_PW,
        metabolites_per_pathway=MPP, time_points=TIME_PTS,
        effect_func=ramp_func, nuisance_fraction=0.2,
        nuisance_effect_func=nuisance_func, annotation_fraction=ANNOTATION_FRAC,
        metabolite_effect_sd=0.5,
    )

    # Perturbation condition
    np.random.seed(seed + 2)
    perturb_func = create_perturbation_effect(4, 4, EFFECT_MAG)
    df_perturb, true_pws_perturb, _ = simulate_longitudinal_data(
        n_subjects=N_SUBJECTS, n_metabolites=N_MET, n_pathways=N_PW,
        metabolites_per_pathway=MPP, time_points=TIME_PTS,
        effect_func=perturb_func, nuisance_fraction=0.2,
        nuisance_effect_func=nuisance_func, annotation_fraction=ANNOTATION_FRAC,
        metabolite_effect_sd=0.5,
    )

    active_pid = list(true_pws_spike.keys())[0]
    active_mets_all = true_pws_spike[active_pid]

    # Pick metabolites that show clear response for each effect type
    display_mets_spike = _select_responsive_mets(df_spike, active_mets_all, n=4)
    display_mets_ramp = _select_trend_mets(df_ramp, true_pws_ramp[active_pid], n=4)
    display_mets_perturb = _select_responsive_mets(
        df_perturb, true_pws_perturb[active_pid],
        effect_window=(4, 8), n=4)

    # Group covariate condition — same seed as spike so pathway members match
    np.random.seed(seed)
    df_group, true_pws_group, _ = simulate_longitudinal_data(
        n_subjects=N_SUBJECTS, n_metabolites=N_MET, n_pathways=N_PW,
        metabolites_per_pathway=MPP, time_points=TIME_PTS,
        effect_func=spike_func, nuisance_fraction=0.2,
        nuisance_effect_func=nuisance_func, annotation_fraction=ANNOTATION_FRAC,
        metabolite_effect_sd=0.5, add_group_covariate=True,
    )

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

    # ---- Figure layout: 2×3 grid ------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_spike   = fig.add_subplot(gs[0, 0])
    ax_ramp    = fig.add_subplot(gs[0, 1])
    ax_perturb = fig.add_subplot(gs[0, 2])
    ax_noise   = fig.add_subplot(gs[1, 0])
    ax_group   = fig.add_subplot(gs[1, 1])
    ax_annot   = fig.add_subplot(gs[1, 2])

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

    # ---- Panel B: Ramp (trapezoid) ---------------------------------------
    sids_ramp = sorted(df_ramp["subject_id"].unique())[:N_DISPLAY_SUBJECTS]
    traj_ramp = _trajectories_for(df_ramp, display_mets_ramp, sids_ramp)
    _plot_trajectories(ax_ramp, traj_ramp, palette_active)
    ax_ramp.axvspan(5, 13, alpha=0.08, color="blue", label="Ramp window")
    ax_ramp.set_title("(B) Active pathway — sustained ramp",
                      fontsize=11, fontweight="bold")
    ax_ramp.set_xlabel("Time")
    ax_ramp.set_ylabel("log(abundance + 1)")
    ax_ramp.legend(fontsize=7, ncol=2, loc="upper left")

    # ---- Panel C: Perturbation -------------------------------------------
    sids_perturb = sorted(df_perturb["subject_id"].unique())[:N_DISPLAY_SUBJECTS]
    traj_perturb = _trajectories_for(df_perturb, display_mets_perturb,
                                     sids_perturb)
    _plot_trajectories(ax_perturb, traj_perturb, palette_active)
    ax_perturb.axvline(4, color="red", ls="--", alpha=0.5,
                       label="Perturbation onset")
    ax_perturb.set_title("(C) Active pathway — acute perturbation",
                         fontsize=11, fontweight="bold")
    ax_perturb.set_xlabel("Time")
    ax_perturb.set_ylabel("log(abundance + 1)")
    ax_perturb.legend(fontsize=7, ncol=2, loc="upper right")

    # ---- Panel D: Nuisance vs null ---------------------------------------
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
    ax_noise.set_title("(D) Nuisance vs. null metabolite", fontsize=11, fontweight="bold")
    ax_noise.set_xlabel("Time")
    ax_noise.set_ylabel("log(abundance + 1)")
    ax_noise.legend(fontsize=8, loc="upper right")

    # ---- Panel E: Group covariate ----------------------------------------
    # Use the first metabolite from Panel A so readers can cross-reference
    mid_g = display_mets_spike[0]
    met_color = "#e41a1c"  # same red as M005 in Panel A
    group_styles = [("-", "Group 0 (full)"), ("--", "Group 1 (50%)")]
    for grp, (ls, label_text) in enumerate(group_styles):
        grp_sids = df_group[df_group["group"] == grp]["subject_id"].unique()
        grp_sids = sorted(grp_sids)[:N_DISPLAY_SUBJECTS]
        sub = df_group[
            (df_group["metabolite_id"] == mid_g) &
            (df_group["subject_id"].isin(grp_sids))
        ].sort_values("time")
        first = True
        for sid in grp_sids:
            s = sub[sub["subject_id"] == sid]
            if s.empty:
                continue
            ax_group.scatter(s["time"], np.log1p(s["value"]),
                             color=met_color, alpha=0.3, s=14, zorder=2)
            ax_group.plot(s["time"], np.log1p(s["true_mu"]),
                          color=met_color, alpha=0.85, lw=2.0,
                          ls=ls, label=label_text if first else None,
                          zorder=3)
            first = False
    ax_group.axvspan(8, 12, alpha=0.10, color="red")
    ax_group.set_title(f"(E) Group covariate — {mid_g}",
                       fontsize=11, fontweight="bold")
    ax_group.set_xlabel("Time")
    ax_group.set_ylabel("log(abundance + 1)")
    ax_group.legend(fontsize=8, loc="upper left")

    # ---- Panel F: Annotation fraction schematic (2×5 grid) ----------------
    n_members = 10
    n_annotated = 7  # exactly 70%
    n_rows, n_cols = 5, 2
    ax_annot.set_xlim(-5, 15)
    ax_annot.set_ylim(-2.0, 7.5)
    ax_annot.axis("off")
    ax_annot.set_title(r"(F) Annotation fraction ($\rho = 0.7$)",
                       fontsize=11, fontweight="bold")

    bw = 2.2   # sub-box width
    bh = 0.75  # sub-box height
    gap_x = 0.4  # horizontal gap between columns
    gap_y = 0.35  # vertical gap between rows
    grid_w = n_cols * bw + (n_cols - 1) * gap_x
    grid_h = n_rows * bh + (n_rows - 1) * gap_y
    x0 = 5.0 - grid_w / 2  # left edge
    y_top = 5.5             # top edge of grid

    # Outer rounded rectangle
    pad = 0.35
    box = mpatches.FancyBboxPatch(
        (x0 - pad, y_top - grid_h - pad), grid_w + 2 * pad, grid_h + 2 * pad,
        boxstyle="round,pad=0.25", facecolor="#f0f0f0",
        edgecolor="#aaaaaa", linewidth=1.5, zorder=0)
    ax_annot.add_patch(box)
    ax_annot.text(5.0, y_top + pad + 0.5,
                  f"True pathway (all {n_members} members)",
                  ha="center", va="bottom", fontsize=9, color="#555555")

    # Place sub-boxes: left column top-to-bottom, then right column
    positions = []  # (cx, cy) for each metabolite
    for idx in range(n_members):
        col = idx // n_rows
        row = idx % n_rows
        cx = x0 + col * (bw + gap_x) + bw / 2
        cy = y_top - row * (bh + gap_y) - bh / 2
        positions.append((cx, cy))
        annotated = idx < n_annotated
        sub = mpatches.FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2), bw, bh,
            boxstyle="round,pad=0.08",
            facecolor="#377eb8" if annotated else "white",
            edgecolor="#377eb8", linewidth=1.5, zorder=2)
        ax_annot.add_patch(sub)
        ax_annot.text(cx, cy, f"M{idx+1}", ha="center", va="center",
                      fontsize=8.5, color="white" if annotated else "#377eb8",
                      fontweight="bold", zorder=3)

    # Bracket left: annotated members (span full grid height for rows 0-6)
    annot_positions = positions[:n_annotated]
    annot_top = max(cy for _, cy in annot_positions) + bh / 2
    annot_bot = min(cy for _, cy in annot_positions) - bh / 2
    mid_annot_y = (annot_top + annot_bot) / 2
    bx = x0 - pad - 0.5
    ax_annot.plot([bx, bx], [annot_top, annot_bot],
                  color="#377eb8", lw=1.3)
    ax_annot.plot([bx, bx + 0.2], [annot_top, annot_top],
                  color="#377eb8", lw=1.3)
    ax_annot.plot([bx, bx + 0.2], [annot_bot, annot_bot],
                  color="#377eb8", lw=1.3)
    ax_annot.text(bx - 0.15, mid_annot_y,
                  "Pathway-aware\nmethods see only\nannotated members\n(filled)",
                  ha="right", va="center", fontsize=7.5, color="#377eb8")

    # Bracket right: withheld members
    with_positions = positions[n_annotated:]
    with_top = max(cy for _, cy in with_positions) + bh / 2
    with_bot = min(cy for _, cy in with_positions) - bh / 2
    mid_with_y = (with_top + with_bot) / 2
    bx_r = x0 + grid_w + pad + 0.5
    ax_annot.plot([bx_r, bx_r], [with_top, with_bot],
                  color="#e41a1c", lw=1.3)
    ax_annot.plot([bx_r, bx_r - 0.2], [with_top, with_top],
                  color="#e41a1c", lw=1.3)
    ax_annot.plot([bx_r, bx_r - 0.2], [with_bot, with_bot],
                  color="#e41a1c", lw=1.3)
    ax_annot.text(bx_r + 0.15, mid_with_y,
                  "Data-driven methods\ncan discover withheld\nmembers (hollow)",
                  ha="left", va="center", fontsize=7.5, color="#e41a1c")

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
        default="figures/simulation_overview.pdf",
        help="Output file path (.pdf or .png)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_figure(args.output, seed=args.seed)
