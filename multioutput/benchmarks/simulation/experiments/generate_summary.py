#!/usr/bin/env python3
"""
generate_summary.py — Summary figures and tables for the simulation benchmark.

Loads benchmark_results.csv files from all experiment conditions and produces:
  1. SNR sweep: pathway methods (Sensitivity, FPR) — spike and linear
  2. SNR sweep: clustering methods (BestJaccard, UnannotatedRecall, Reconstruction_MSE)
  3. Annotation fraction sweep: pathway and clustering metrics
  4. Timing comparison across conditions and methods
  5. Group covariate condition (with PAL)
  6. Summary CSV table (mean ± SE per method per condition)

Usage:
    python3 simulation/experiments/generate_summary.py

Outputs saved to:
    simulation/experiments/output/summary/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, "output")
OUT_DIR = os.path.join(OUTPUT_BASE, "summary")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Method metadata ───────────────────────────────────────────────────────────

CLUSTERING_METHODS = ["MOGP", "WGCNA", "DPGP", "MEFISTO", "timeOmics"]
PATHWAY_METHODS    = ["MOGP_ORA", "MOGP_GSEA", "MEBA", "LMM_ORA", "LMM_GSEA", "PAL"]

METHOD_LABELS = {
    "MOGP":      "MOGP",
    "MOGP_ORA":  "MOGP+ORA",
    "MOGP_GSEA": "MOGP+GSEA",
    "WGCNA":     "WGCNA",
    "MEFISTO":   "MEFISTO",
    "DPGP":      "DPGP",
    "timeOmics": "timeOmics",
    "LMM_ORA":   "LMM+ORA",
    "LMM_GSEA":  "LMM+GSEA",
    "MEBA":      "MEBA",
    "PAL":       "PAL",
}

METHOD_COLORS = {
    "MOGP":      "#e41a1c",
    "MOGP_ORA":  "#e41a1c",
    "MOGP_GSEA": "#ff6666",
    "WGCNA":     "#377eb8",
    "MEFISTO":   "#4daf4a",
    "DPGP":      "#984ea3",
    "timeOmics": "#ff7f00",
    "LMM_ORA":   "#a65628",
    "LMM_GSEA":  "#f781bf",
    "MEBA":      "#999999",
    "PAL":       "#66c2a5",
}

METHOD_MARKER = {
    "MOGP":      "o",
    "MOGP_ORA":  "o",
    "MOGP_GSEA": "h",
    "WGCNA":     "s",
    "MEFISTO":   "^",
    "DPGP":      "D",
    "timeOmics": "P",
    "LMM_ORA":   "v",
    "LMM_GSEA":  "<",
    "MEBA":      "X",
    "PAL":       "*",
}

METHOD_LW = {m: 2.5 if "MOGP" in m else 1.5 for m in METHOD_LABELS}
METHOD_LS = {m: "-" if "MOGP" in m else "--" for m in METHOD_LABELS}

SNR_ORDER = ["easy", "medium", "difficult"]
SNR_LABELS = {"easy": "Easy", "medium": "Medium", "difficult": "Difficult"}

EFFECT_LABELS = {"spike": "Spike", "linear": "Linear", "perturbation": "Perturbation"}

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_csv(path, snr_level="", annotation_fraction=None, has_group_covariate=False):
    df = pd.read_csv(path)
    df["snr_level"] = snr_level
    df["annotation_fraction"] = annotation_fraction
    df["has_group_covariate"] = has_group_covariate
    return df


def load_all_data():
    frames = []

    # SNR sweep
    for snr in SNR_ORDER:
        for effect in ["spike", "linear", "perturbation"]:
            p = os.path.join(OUTPUT_BASE, f"results_{snr}", effect, "benchmark_results.csv")
            if os.path.exists(p):
                frames.append(_load_csv(p, snr_level=snr))

    # Annotation sweep (medium SNR)
    for frac in ["0.3", "0.5", "0.9"]:
        for effect in ["spike", "linear", "perturbation"]:
            p = os.path.join(OUTPUT_BASE, "results_annotation", f"annot_{frac}", effect, "benchmark_results.csv")
            if os.path.exists(p):
                frames.append(_load_csv(p, annotation_fraction=float(frac)))

    # Group covariate (medium SNR)
    for effect in ["spike", "linear", "perturbation"]:
        p = os.path.join(OUTPUT_BASE, "results_group_covariate", effect, "benchmark_results.csv")
        if os.path.exists(p):
            frames.append(_load_csv(p, snr_level="medium", has_group_covariate=True))

    df = pd.concat(frames, ignore_index=True)
    df["snr_level"] = pd.Categorical(df["snr_level"], categories=[""] + SNR_ORDER, ordered=False)
    return df


# ── Aggregate helper ──────────────────────────────────────────────────────────

def agg(df, col):
    """Return mean and SEM of col, dropping NaN."""
    vals = df[col].dropna()
    if vals.empty:
        return np.nan, np.nan
    return vals.mean(), vals.sem()


# ── Figure helpers ────────────────────────────────────────────────────────────

def _style_ax(ax, xlabel=None, ylabel=None, title=None, ylim=None, legend=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    if ylim is not None:
        ax.set_ylim(ylim)
    if legend:
        ax.legend(fontsize=7, framealpha=0.7, loc="best")


def _plot_snr_lines(ax, df_snr, methods, metric_col, effect, ylabel, flag_missing=None):
    """Line plot of mean ± SEM across SNR levels for given methods."""
    for m in methods:
        col = f"{m}_{metric_col}"
        if col not in df_snr.columns:
            continue
        xs, ys, errs = [], [], []
        for snr in SNR_ORDER:
            sub = df_snr[(df_snr["snr_level"] == snr) & (df_snr["effect_type"] == effect)]
            mn, se = agg(sub, col)
            xs.append(snr)
            ys.append(mn)
            errs.append(se)
        xs_num = list(range(len(SNR_ORDER)))
        label = METHOD_LABELS.get(m, m)
        color = METHOD_COLORS.get(m, "gray")
        ls = METHOD_LS.get(m, "--")
        lw = METHOD_LW.get(m, 1.5)
        mk = METHOD_MARKER.get(m, "o")
        ax.errorbar(xs_num, ys, yerr=errs, label=label, color=color,
                    linestyle=ls, linewidth=lw, marker=mk, markersize=5,
                    capsize=3, capthick=1)
    ax.set_xticks(range(len(SNR_ORDER)))
    ax.set_xticklabels([SNR_LABELS[s] for s in SNR_ORDER])
    _style_ax(ax, xlabel="SNR Level", ylabel=ylabel)


# ── Figure 1: SNR sweep — Pathway methods ────────────────────────────────────

def fig_snr_pathway(df):
    snr_df = df[(df["snr_level"].isin(SNR_ORDER)) & (~df["has_group_covariate"])].copy()

    pathway_methods = ["MOGP_ORA", "MOGP_GSEA", "MEBA", "LMM_ORA", "LMM_GSEA"]
    metrics = [
        ("Sensitivity",       "Sensitivity (TPR)",   (0, 1.05)),
        ("FPR",               "False Positive Rate",  (0, 1.05)),
        ("Reconstruction_MSE","Reconstruction MSE",   None),
    ]

    effects = [e for e in ["linear", "spike", "perturbation"]
               if e in snr_df["effect_type"].unique()]
    n_effects = len(effects)
    fig, axes = plt.subplots(n_effects, 3, figsize=(18, 4 * n_effects), sharex=True)
    if n_effects == 1:
        axes = [axes]
    for row, effect in enumerate(effects):
        for col, (metric, ylabel, ylim) in enumerate(metrics):
            ax = axes[row][col]
            _plot_snr_lines(ax, snr_df, pathway_methods, metric, effect, ylabel)
            if ylim:
                ax.set_ylim(ylim)
            ax.set_title(f"{EFFECT_LABELS[effect]} — {ylabel}", fontsize=11, fontweight="bold")
            if row == 1:
                ax.set_xlabel("SNR Level", fontsize=10)
            else:
                ax.set_xlabel("")

    fig.suptitle("Pathway Detection Performance Across SNR Levels", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_snr_pathway.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── Figure 2: SNR sweep — Clustering methods ─────────────────────────────────

def fig_snr_clustering(df):
    snr_df = df[(df["snr_level"].isin(SNR_ORDER)) & (~df["has_group_covariate"])].copy()

    clust_methods = ["MOGP", "WGCNA", "DPGP", "MEFISTO", "timeOmics"]
    metrics = [
        ("ARI",               "ARI",                           None),
        ("BestJaccard",       "Best Jaccard (active pathway)", (0, 1.0)),
        ("BestPrecision",     "Best Precision (active pathway)", (0, 1.0)),
        ("UnannotatedRecall", "Unannotated Metabolite Recall", (0, 1.05)),
        ("Reconstruction_MSE","Reconstruction MSE",            None),
    ]

    effects = [e for e in ["linear", "spike", "perturbation"]
               if e in snr_df["effect_type"].unique()]
    n_effects = len(effects)
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_effects, n_metrics, figsize=(5 * n_metrics, 4 * n_effects), sharex=True)
    if n_effects == 1:
        axes = [axes]
    for row, effect in enumerate(effects):
        for col, (metric, ylabel, ylim) in enumerate(metrics):
            ax = axes[row][col]
            _plot_snr_lines(ax, snr_df, clust_methods, metric, effect, ylabel)
            if ylim:
                ax.set_ylim(ylim)
            ax.set_title(f"{EFFECT_LABELS[effect]} — {ylabel}", fontsize=10, fontweight="bold")
            if row == 1:
                ax.set_xlabel("SNR Level", fontsize=10)

    fig.suptitle("Clustering Performance Across SNR Levels", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_snr_clustering.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── Figure 3: Annotation sweep ────────────────────────────────────────────────

def fig_annotation_sweep(df):
    # Include medium condition (annot=0.7) in the sweep
    medium_df = df[
        (df["snr_level"] == "medium") &
        (~df["has_group_covariate"]) &
        (df["annotation_fraction"].isna())
    ].copy()
    medium_df["annotation_fraction"] = 0.7

    sweep_df = df[df["annotation_fraction"].notna()].copy()
    combined = pd.concat([sweep_df, medium_df], ignore_index=True)
    fracs = sorted(combined["annotation_fraction"].dropna().unique())

    pathway_methods = ["MOGP_ORA", "MOGP_GSEA", "MEBA", "LMM_ORA", "LMM_GSEA"]
    clust_methods   = ["MOGP", "WGCNA", "DPGP", "MEFISTO", "timeOmics"]

    metrics_path = [
        ("Sensitivity",       "Sensitivity (TPR)",  (0, 1.05)),
        ("FPR",               "False Positive Rate", (0, 1.05)),
        ("Reconstruction_MSE","Reconstruction MSE",  None),
    ]
    metrics_clust = [
        ("BestJaccard",       "Best Jaccard",           (0, 1.0)),
        ("BestPrecision",     "Best Precision",          (0, 1.0)),
        ("UnannotatedRecall", "Unannotated Recall",      (0, 1.05)),
        ("Reconstruction_MSE","Reconstruction MSE",     None),
    ]

    n_path = len(metrics_path)
    n_clust = len(metrics_clust)
    n_cols = n_path + n_clust
    for effect in ["linear", "spike", "perturbation"]:
        edf = combined[combined["effect_type"] == effect]
        if edf.empty:
            continue
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))

        for ax, (metric, ylabel, ylim) in zip(axes[:n_path], metrics_path):
            for m in pathway_methods:
                col = f"{m}_{metric}"
                if col not in edf.columns:
                    continue
                xs, ys, errs = [], [], []
                for frac in fracs:
                    sub = edf[edf["annotation_fraction"] == frac]
                    mn, se = agg(sub, col)
                    xs.append(frac)
                    ys.append(mn)
                    errs.append(se)
                ax.errorbar(xs, ys, yerr=errs,
                            label=METHOD_LABELS.get(m, m),
                            color=METHOD_COLORS.get(m, "gray"),
                            linestyle=METHOD_LS.get(m, "--"),
                            linewidth=METHOD_LW.get(m, 1.5),
                            marker=METHOD_MARKER.get(m, "o"),
                            markersize=5, capsize=3, capthick=1)
            ax.axvline(0.7, color="gray", linestyle=":", linewidth=1, alpha=0.6)
            ax.set_xlim(0.25, 0.95)
            if ylim:
                ax.set_ylim(ylim)
            _style_ax(ax, xlabel="Annotation Fraction", ylabel=ylabel, title=ylabel)

        for ax, (metric, ylabel, ylim) in zip(axes[n_path:], metrics_clust):
            for m in clust_methods:
                col = f"{m}_{metric}"
                if col not in edf.columns:
                    continue
                xs, ys, errs = [], [], []
                for frac in fracs:
                    sub = edf[edf["annotation_fraction"] == frac]
                    mn, se = agg(sub, col)
                    xs.append(frac)
                    ys.append(mn)
                    errs.append(se)
                ax.errorbar(xs, ys, yerr=errs,
                            label=METHOD_LABELS.get(m, m),
                            color=METHOD_COLORS.get(m, "gray"),
                            linestyle=METHOD_LS.get(m, "--"),
                            linewidth=METHOD_LW.get(m, 1.5),
                            marker=METHOD_MARKER.get(m, "o"),
                            markersize=5, capsize=3, capthick=1)
            ax.axvline(0.7, color="gray", linestyle=":", linewidth=1, alpha=0.6)
            ax.set_xlim(0.25, 0.95)
            if ylim:
                ax.set_ylim(ylim)
            _style_ax(ax, xlabel="Annotation Fraction", ylabel=ylabel, title=ylabel)

        fig.suptitle(f"Annotation Fraction Sweep — {EFFECT_LABELS[effect]} Effect\n"
                     f"(dotted line = default 0.7; medium SNR)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"fig3_annotation_sweep_{effect}.png")
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


# ── Figure 4: Timing ──────────────────────────────────────────────────────────

def fig_timing(df):
    snr_df = df[(df["snr_level"].isin(SNR_ORDER)) & (~df["has_group_covariate"])].copy()

    time_cols = {
        "MOGP":      "MOGP_Time",
        "MOGP+ORA":  "MOGP_ORA_Time",
        "MOGP+GSEA": "MOGP_GSEA_Time",
        "WGCNA":     "WGCNA_Time",
        "MEFISTO":   "MEFISTO_Time",
        "DPGP":      "DPGP_Time",
        "timeOmics": "timeOmics_Time",
        "LMM+ORA":   "LMM_ORA_Time",
        "LMM+GSEA":  "LMM_GSEA_Time",
        "MEBA":      "MEBA_Time",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, snr in zip(axes, SNR_ORDER):
        sub = snr_df[snr_df["snr_level"] == snr]
        means, labels, colors = [], [], []
        for label, col in time_cols.items():
            if col not in sub.columns:
                continue
            m = sub[col].dropna().mean()
            base_method = col.replace("_Time", "").replace("_GSEA", "_GSEA")
            # map label to a base method key for color
            color_key = next((k for k in METHOD_COLORS if METHOD_LABELS.get(k, k) == label), None)
            if color_key is None:
                color_key = next((k for k in METHOD_COLORS if k in label.replace("+", "_")), None)
            color = METHOD_COLORS.get(color_key, "gray") if color_key else "gray"
            means.append(m)
            labels.append(label)
            colors.append(color)

        # Sort by mean time descending
        order = np.argsort(means)[::-1]
        sorted_means  = [means[i] for i in order]
        sorted_labels = [labels[i] for i in order]
        sorted_colors = [colors[i] for i in order]

        bars = ax.barh(range(len(sorted_means)), sorted_means, color=sorted_colors, edgecolor="white")
        ax.set_yticks(range(len(sorted_labels)))
        ax.set_yticklabels(sorted_labels, fontsize=9)
        ax.set_xlabel("Mean Runtime (s)", fontsize=10)
        ax.set_title(f"{SNR_LABELS[snr]} SNR", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.suptitle("Mean Runtime per Method Across SNR Conditions (spike + linear pooled)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_timing.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── Figure 5: Group covariate with PAL ───────────────────────────────────────

def fig_group_covariate(df):
    gc_df  = df[df["has_group_covariate"]].copy()
    if gc_df.empty:
        print("No group covariate data — skipping fig5.")
        return

    pathway_methods_gc = ["MOGP_ORA", "MOGP_GSEA", "MEBA", "PAL", "LMM_ORA", "LMM_GSEA"]
    clustering_methods_gc = ["MOGP", "WGCNA", "DPGP", "MEFISTO", "timeOmics"]

    pathway_metrics = [
        ("Sensitivity", "Sensitivity (TPR)",  (0, 1.05)),
        ("FPR",         "False Positive Rate", (0, 1.05)),
    ]
    clustering_metrics = [
        ("ARI",         "ARI",         None),
        ("BestJaccard", "Best Jaccard", (0, 1.0)),
        ("NumModules",  "Num Modules",  None),
    ]

    effects = [e for e in ["linear", "spike", "perturbation"]
               if e in gc_df["effect_type"].unique()]
    n_effects = len(effects)
    n_cols = len(pathway_metrics) + len(clustering_metrics)

    fig, axes = plt.subplots(n_effects, n_cols, figsize=(4.5 * n_cols, 4.5 * n_effects))
    if n_effects == 1:
        axes = [axes]

    for row, effect in enumerate(effects):
        edf = gc_df[gc_df["effect_type"] == effect]

        # Pathway metrics
        for col, (metric, ylabel, ylim) in enumerate(pathway_metrics):
            ax = axes[row][col]
            ys, yerrs, labels, colors = [], [], [], []
            for m in pathway_methods_gc:
                mcol = f"{m}_{metric}"
                if mcol not in edf.columns:
                    continue
                mn, se = agg(edf, mcol)
                ys.append(mn); yerrs.append(se)
                labels.append(METHOD_LABELS.get(m, m))
                colors.append(METHOD_COLORS.get(m, "gray"))
            ax.bar(range(len(ys)), ys, yerr=yerrs, color=colors, capsize=4,
                   edgecolor="white", error_kw={"elinewidth": 1.5})
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
            if ylim:
                ax.set_ylim(ylim)
            if col == 0:
                ax.set_ylabel(f"{EFFECT_LABELS[effect]}\n\n{ylabel}", fontsize=10)
            if row == 0:
                ax.set_title(ylabel, fontsize=11, fontweight="bold")
            _style_ax(ax, legend=False)

        # Clustering metrics
        for i, (metric, ylabel, ylim) in enumerate(clustering_metrics):
            col = len(pathway_metrics) + i
            ax = axes[row][col]
            ys, yerrs, labels, colors = [], [], [], []
            for m in clustering_methods_gc:
                mcol = f"{m}_{metric}"
                if mcol not in edf.columns:
                    continue
                mn, se = agg(edf, mcol)
                ys.append(mn); yerrs.append(se)
                labels.append(METHOD_LABELS.get(m, m))
                colors.append(METHOD_COLORS.get(m, "gray"))
            ax.bar(range(len(ys)), ys, yerr=yerrs, color=colors, capsize=4,
                   edgecolor="white", error_kw={"elinewidth": 1.5})
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
            if ylim:
                ax.set_ylim(ylim)
            if row == 0:
                ax.set_title(ylabel, fontsize=11, fontweight="bold")
            _style_ax(ax, legend=False)

    # Divider between pathway and clustering sections
    for row in range(n_effects):
        axes[row][len(pathway_metrics)].spines["left"].set_linewidth(2.5)

    fig.suptitle("Group Covariate Condition — Pathway Detection & Clustering Performance\n"
                 "(Medium SNR, effect_magnitude=6.0, group 1 receives 50% of temporal effect)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_group_covariate.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def write_summary_table(df):
    rows = []

    # SNR sweep (excluding group covariate)
    snr_df = df[(df["snr_level"].isin(SNR_ORDER)) & (~df["has_group_covariate"])]
    for snr in SNR_ORDER:
        for effect in ["linear", "spike", "perturbation"]:
            sub = snr_df[(snr_df["snr_level"] == snr) & (snr_df["effect_type"] == effect)]
            if sub.empty:
                continue
            n = len(sub)
            for m in CLUSTERING_METHODS:
                for metric, col in [("ARI", f"{m}_ARI"),
                                     ("BestJaccard", f"{m}_BestJaccard"),
                                     ("BestPrecision", f"{m}_BestPrecision"),
                                     ("UnannotatedRecall", f"{m}_UnannotatedRecall"),
                                     ("NumModules", f"{m}_NumModules"),
                                     ("Reconstruction_MSE", f"{m}_Reconstruction_MSE")]:
                    if col not in sub.columns:
                        continue
                    mn, se = agg(sub, col)
                    rows.append({"condition": snr, "effect": effect,
                                 "method": METHOD_LABELS.get(m, m), "metric": metric,
                                 "mean": round(mn, 4) if not np.isnan(mn) else np.nan,
                                 "se": round(se, 4) if not np.isnan(se) else np.nan,
                                 "n": n})
            for m in PATHWAY_METHODS:
                for metric, col in [("Sensitivity", f"{m}_Sensitivity"),
                                     ("FPR", f"{m}_FPR"),
                                     ("Reconstruction_MSE", f"{m}_Reconstruction_MSE")]:
                    if col not in sub.columns:
                        continue
                    mn, se = agg(sub, col)
                    rows.append({"condition": snr, "effect": effect,
                                 "method": METHOD_LABELS.get(m, m), "metric": metric,
                                 "mean": round(mn, 4) if not np.isnan(mn) else np.nan,
                                 "se": round(se, 4) if not np.isnan(se) else np.nan,
                                 "n": n})

    # Annotation sweep
    medium_df = df[(df["snr_level"] == "medium") & (~df["has_group_covariate"]) & (df["annotation_fraction"].isna())].copy()
    medium_df["annotation_fraction"] = 0.7
    sweep_df = pd.concat([df[df["annotation_fraction"].notna()], medium_df], ignore_index=True)
    for frac in sorted(sweep_df["annotation_fraction"].dropna().unique()):
        for effect in ["linear", "spike", "perturbation"]:
            sub = sweep_df[(sweep_df["annotation_fraction"] == frac) & (sweep_df["effect_type"] == effect)]
            if sub.empty:
                continue
            n = len(sub)
            cond_label = f"annot_{frac}"
            for m in PATHWAY_METHODS:
                for metric, col in [("Sensitivity", f"{m}_Sensitivity"), ("FPR", f"{m}_FPR")]:
                    if col not in sub.columns:
                        continue
                    mn, se = agg(sub, col)
                    rows.append({"condition": cond_label, "effect": effect,
                                 "method": METHOD_LABELS.get(m, m), "metric": metric,
                                 "mean": round(mn, 4) if not np.isnan(mn) else np.nan,
                                 "se": round(se, 4) if not np.isnan(se) else np.nan,
                                 "n": n})

    # Group covariate
    gc_df = df[df["has_group_covariate"]]
    for effect in ["linear", "spike", "perturbation"]:
        sub = gc_df[gc_df["effect_type"] == effect]
        if sub.empty:
            continue
        n = len(sub)
        for m in PATHWAY_METHODS:
            for metric, col in [("Sensitivity", f"{m}_Sensitivity"), ("FPR", f"{m}_FPR")]:
                if col not in sub.columns:
                    continue
                mn, se = agg(sub, col)
                rows.append({"condition": "group_covariate", "effect": effect,
                             "method": METHOD_LABELS.get(m, m), "metric": metric,
                             "mean": round(mn, 4) if not np.isnan(mn) else np.nan,
                             "se": round(se, 4) if not np.isnan(se) else np.nan,
                             "n": n})

    out = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "summary_table.csv")
    out.to_csv(path, index=False)
    print(f"Saved {path}")
    return out


# ── Figure: Main manuscript simulation figure ────────────────────────────────

# Publication rcParams applied locally within fig_main_simulation
_PUB_RC = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "errorbar.capsize": 2.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Focused method sets for the main figure
_MAIN_PATHWAY_METHODS = ["MOGP_ORA", "MOGP_GSEA", "LMM_ORA", "LMM_GSEA", "MEBA"]
_MAIN_CLUST_METHODS   = ["MOGP", "MEFISTO", "DPGP", "WGCNA"]

# Line styles: solid for "our" methods, dashed for baselines
_MAIN_LS = {
    "MOGP_ORA":  "-",
    "MOGP_GSEA": "-",
    "LMM_ORA":   "--",
    "LMM_GSEA":  "--",
    "MEBA":      ":",
    "MOGP":      "-",
    "MEFISTO":   "--",
    "DPGP":      "--",
    "WGCNA":     ":",
}
_MAIN_LW = {
    "MOGP_ORA":  2.0,
    "MOGP_GSEA": 2.0,
    "LMM_ORA":   1.2,
    "LMM_GSEA":  1.2,
    "MEBA":      1.2,
    "MOGP":      2.0,
    "MEFISTO":   1.2,
    "DPGP":      1.2,
    "WGCNA":     1.2,
}


def fig_main_simulation(df):
    """
    Publication-quality 4×3 main simulation figure.

    Rows: Sensitivity (pathway) | FPR (pathway) | Best Jaccard (clustering) | Reconstruction MSE
    Columns: Spike | Linear | Perturbation

    Saved to the summary output directory as PNG (300 DPI).
    """
    import matplotlib as mpl

    snr_df = df[(df["snr_level"].isin(SNR_ORDER)) & (~df["has_group_covariate"])].copy()
    effects = ["spike", "linear", "perturbation"]
    effect_titles = {"spike": "Spike", "linear": "Linear", "perturbation": "Perturbation"}

    rows = [
        ("pathway", "Sensitivity",       "Sensitivity (TPR)",   _MAIN_PATHWAY_METHODS),
        ("pathway", "FPR",               "False Positive Rate", _MAIN_PATHWAY_METHODS),
        ("clust",   "BestJaccard",       "Best Jaccard",        _MAIN_CLUST_METHODS),
        ("clust",   "Reconstruction_MSE","Reconstruction MSE",  _MAIN_CLUST_METHODS),
    ]

    with mpl.rc_context(_PUB_RC):
        fig, axes = plt.subplots(4, 3, figsize=(8.5, 7.5), sharex=True, sharey="row")
        xs_num = list(range(len(SNR_ORDER)))

        for row_i, (kind, metric, ylabel, methods) in enumerate(rows):
            for col_i, effect in enumerate(effects):
                ax = axes[row_i][col_i]
                for m in methods:
                    col = f"{m}_{metric}"
                    if col not in snr_df.columns:
                        continue
                    ys, errs = [], []
                    for snr in SNR_ORDER:
                        sub = snr_df[(snr_df["snr_level"] == snr) &
                                     (snr_df["effect_type"] == effect)]
                        mn, se = agg(sub, col)
                        ys.append(mn); errs.append(se)
                    ax.errorbar(
                        xs_num, ys, yerr=errs,
                        label=METHOD_LABELS.get(m, m),
                        color=METHOD_COLORS.get(m, "gray"),
                        linestyle=_MAIN_LS.get(m, "--"),
                        linewidth=_MAIN_LW.get(m, 1.2),
                        marker=METHOD_MARKER.get(m, "o"),
                        capthick=1,
                    )
                ax.set_xticks(xs_num)
                ax.grid(axis="y", linestyle=":", alpha=0.35)
                # Column titles only on top row
                if row_i == 0:
                    ax.set_title(effect_titles[effect], fontweight="bold")
                # Row labels only on leftmost column
                if col_i == 0:
                    ax.set_ylabel(ylabel)
                # X-axis tick labels only on bottom row
                if row_i == 3:
                    ax.set_xticklabels([SNR_LABELS[s] for s in SNR_ORDER])

                # Legend: top-right panel for pathway rows, top-right for clustering row
                show_legend = (col_i == 2) and (row_i in (0, 2))
                if show_legend:
                    ax.legend(loc="upper right", framealpha=0.8)

        # Horizontal divider between pathway and clustering row groups
        for col_i in range(3):
            axes[2][col_i].spines["top"].set_visible(True)
            axes[2][col_i].spines["top"].set_linewidth(1.0)
            axes[2][col_i].spines["top"].set_linestyle("--")
            axes[2][col_i].spines["top"].set_color("#aaaaaa")

        # Row-group annotations in the figure margin
        fig.text(0.01, 0.76, "Pathway\ndetection", va="center", ha="left",
                 fontsize=8, color="#555555", style="italic",
                 rotation=90)
        fig.text(0.01, 0.25, "Clustering\nrecovery", va="center", ha="left",
                 fontsize=8, color="#555555", style="italic",
                 rotation=90)

        fig.supxlabel("SNR Condition", fontsize=9)
        plt.tight_layout(rect=[0.04, 0.03, 1, 1])

        os.makedirs(OUT_DIR, exist_ok=True)
        out_png = os.path.join(OUT_DIR, "fig_sim_main.png")
        plt.savefig(out_png)
        plt.close()
        print(f"Saved {out_png}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading all benchmark data...")
    df = load_all_data()
    total = len(df)
    snr_count = df[df["snr_level"].isin(SNR_ORDER)].shape[0]
    annot_count = df[df["annotation_fraction"].notna()].shape[0]
    gc_count = df[df["has_group_covariate"]].shape[0]
    print(f"  Total replicates: {total}")
    print(f"  SNR sweep: {snr_count}  |  Annotation sweep: {annot_count}  |  Group covariate: {gc_count}")
    print()

    print("Generating figures...")
    fig_snr_pathway(df)
    fig_snr_clustering(df)
    fig_annotation_sweep(df)
    fig_timing(df)
    fig_group_covariate(df)
    fig_main_simulation(df)

    print("\nWriting summary table...")
    summary = write_summary_table(df)
    print(f"  {len(summary)} rows written.")

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
