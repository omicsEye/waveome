#!/usr/bin/env python3
"""
aggregate_results.py — Cross-condition summary plots and tables for the benchmark.

Reads all benchmark_results.csv files under a base output directory, then produces:
  1. Annotation sweep line plots (primary figures)
  2. SNR sweep line plots (supplemental)
  3. Group covariate bar plot (supplemental)
  4. Summary CSV table (mean ± SD per method per condition)

Usage:
    python3 simulation/experiments/aggregate_results.py <base_output_dir> [--out_dir <dir>]

Example:
    python3 simulation/experiments/aggregate_results.py \
        simulation/experiments/output/final_benchmark \
        --out_dir paper_figures/
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Method metadata ───────────────────────────────────────────────────────────

# Clustering methods: evaluated by ARI, Jaccard, UnannotatedRecall, MSE
CLUSTERING_METHODS = ["MOGP", "WGCNA", "MEFISTO", "DPGP", "timeOmics"]

# Pathway methods
PATHWAY_METHODS = ["MOGP_ORA", "MOGP_GSEA", "LMM_ORA", "LMM_GSEA", "MEBA", "PAL"]

# Display names for plots
METHOD_LABELS = {
    "MOGP": "MOGP",
    "MOGP_ORA": "MOGP+ORA",
    "MOGP_GSEA": "MOGP+GSEA",
    "WGCNA": "WGCNA",
    "MEFISTO": "MEFISTO",
    "DPGP": "DPGP",
    "timeOmics": "timeOmics",
    "LMM_ORA": "LMM+ORA",
    "LMM_GSEA": "LMM+GSEA",
    "MEBA": "MEBA",
    "PAL": "PAL",
}

# Color palette: MOGP variants always highlighted
METHOD_COLORS = {
    "MOGP":       "#e41a1c",   # red
    "MOGP_ORA":   "#e41a1c",   # red (same underlying model)
    "MOGP_GSEA":  "#ff6666",   # lighter red
    "WGCNA":     "#377eb8",
    "MEFISTO":   "#4daf4a",
    "DPGP":      "#984ea3",
    "timeOmics": "#ff7f00",
    "LMM_ORA":   "#a65628",
    "LMM_GSEA":  "#f781bf",
    "MEBA":      "#999999",
    "PAL":       "#66c2a5",
}

# MOGP uses solid line, others dashed
METHOD_LINESTYLE = {m: "-" if "MOGP" in m else "--" for m in METHOD_LABELS}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all(base_dir: str) -> pd.DataFrame:
    """Load all benchmark_results.csv files under base_dir, inferring condition metadata."""
    frames = []
    for csv_path in sorted(glob.glob(os.path.join(base_dir, "**", "benchmark_results.csv"), recursive=True)):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: could not read {csv_path}: {e}")
            continue

        # Infer annotation_fraction and snr_level from path
        rel = os.path.relpath(csv_path, base_dir)
        parts = rel.replace("\\", "/").split("/")

        df["source_path"] = csv_path

        # Use condition_label column if present (written by main.py --condition_label)
        if "condition_label" not in df.columns:
            df["condition_label"] = ""

        # Parse annotation fraction from label like "annot_0.7"
        df["annotation_fraction"] = np.nan
        for p in parts:
            if p.startswith("annot_"):
                try:
                    df["annotation_fraction"] = float(p.replace("annot_", ""))
                except ValueError:
                    pass

        # Parse SNR level
        df["snr_level"] = ""
        for p in parts:
            if p in ("snr_easy", "snr_medium", "snr_difficult"):
                df["snr_level"] = p.replace("snr_", "")

        # Flag group covariate condition
        df["has_group_covariate"] = any("group_covariate" in p for p in parts)

        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No benchmark_results.csv files found under {base_dir}")

    return pd.concat(frames, ignore_index=True)


# ── Metric helpers ────────────────────────────────────────────────────────────

def melt_metric(df: pd.DataFrame, suffix: str, methods: list, value_name: str) -> pd.DataFrame:
    """Melt columns ending with suffix for the given methods into long format."""
    cols = [f"{m}{suffix}" for m in methods if f"{m}{suffix}" in df.columns]
    if not cols:
        return pd.DataFrame()
    id_vars = [c for c in ["run_id", "effect_type", "condition_label",
                            "annotation_fraction", "snr_level", "has_group_covariate"]
               if c in df.columns]
    melted = df.melt(id_vars=id_vars, value_vars=cols, var_name="Method", value_name=value_name)
    melted["Method"] = melted["Method"].str.replace(suffix, "", regex=False)
    melted["Method_label"] = melted["Method"].map(METHOD_LABELS).fillna(melted["Method"])
    return melted


# ── Figure 1: Annotation Sweep (primary) ─────────────────────────────────────

def plot_annotation_sweep(df: pd.DataFrame, out_dir: str):
    """Line plots across annotation fractions for key metrics."""
    sweep = df[df["annotation_fraction"].notna()].copy()
    if sweep.empty:
        print("No annotation sweep data found — skipping.")
        return

    metrics = [
        ("_BestJaccard",        CLUSTERING_METHODS, "Best Jaccard (active pathway)",   "jaccard"),
        ("_BestPrecision",      CLUSTERING_METHODS, "Best Precision (active pathway)",  "precision"),
        ("_UnannotatedRecall",   CLUSTERING_METHODS, "Unannotated Metabolite Recall",   "recall"),
        ("_Reconstruction_MSE",  CLUSTERING_METHODS, "Reconstruction MSE (log scale)",  "mse_clust"),
        ("_Sensitivity",         PATHWAY_METHODS,    "Pathway Detection Sensitivity",   "sensitivity"),
        ("_FPR",                 PATHWAY_METHODS,    "Pathway False Positive Rate",      "fpr"),
    ]

    for effect in sweep["effect_type"].unique():
        edf = sweep[sweep["effect_type"] == effect]
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)
        if n_metrics == 1:
            axes = [axes]

        for ax, (suffix, methods, ylabel, _) in zip(axes, metrics):
            melted = melt_metric(edf, suffix, methods, ylabel)
            if melted.empty:
                ax.set_visible(False)
                continue
            agg = (melted.groupby(["annotation_fraction", "Method"])[ylabel]
                   .agg(["mean", "sem"]).reset_index())
            for method in agg["Method"].unique():
                mdf = agg[agg["Method"] == method].sort_values("annotation_fraction")
                label = METHOD_LABELS.get(method, method)
                color = METHOD_COLORS.get(method, "gray")
                ls = METHOD_LINESTYLE.get(method, "--")
                lw = 2.5 if "MOGP" in method else 1.5
                ax.plot(mdf["annotation_fraction"], mdf["mean"],
                        label=label, color=color, linestyle=ls, linewidth=lw, marker="o", markersize=5)
                ax.fill_between(mdf["annotation_fraction"],
                                mdf["mean"] - mdf["sem"],
                                mdf["mean"] + mdf["sem"],
                                alpha=0.15, color=color)
            ax.set_xlabel("Annotation Fraction")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend(fontsize=7)
            ax.set_xlim(0.25, 0.95)

        fig.suptitle(f"Annotation Sweep — {effect} effect", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"annotation_sweep_{effect}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")


# ── Figure 2: SNR Sweep (supplemental) ───────────────────────────────────────

SNR_ORDER = ["easy", "medium", "difficult"]

def plot_snr_sweep(df: pd.DataFrame, out_dir: str):
    """Boxplots by SNR level for key clustering metrics."""
    sweep = df[df["snr_level"] != ""].copy()
    if sweep.empty:
        print("No SNR sweep data found — skipping.")
        return

    sweep["snr_level"] = pd.Categorical(sweep["snr_level"], categories=SNR_ORDER, ordered=True)

    for effect in sweep["effect_type"].unique():
        edf = sweep[sweep["effect_type"] == effect]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        configs = [
            ("_BestJaccard",       CLUSTERING_METHODS, "Best Jaccard"),
            ("_UnannotatedRecall", CLUSTERING_METHODS, "Unannotated Recall"),
            ("_Sensitivity",       PATHWAY_METHODS,    "Pathway Sensitivity"),
        ]
        for ax, (suffix, methods, ylabel) in zip(axes, configs):
            melted = melt_metric(edf, suffix, methods, ylabel)
            if melted.empty:
                ax.set_visible(False)
                continue
            palette = {METHOD_LABELS.get(m, m): METHOD_COLORS.get(m, "gray") for m in methods}
            sns.boxplot(data=melted, x="snr_level", y=ylabel, hue="Method_label",
                        palette=palette, ax=ax, order=SNR_ORDER)
            ax.set_xlabel("SNR Level")
            ax.set_title(ylabel)
            ax.legend(fontsize=7, title=None)

        fig.suptitle(f"SNR Sweep — {effect} effect", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"snr_sweep_{effect}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")


# ── Figure 3: Group Covariate (supplemental) ─────────────────────────────────

def plot_group_covariate(df: pd.DataFrame, out_dir: str):
    """Compare MSE with and without group covariate for clustering methods."""
    gc = df[df["has_group_covariate"] == True]
    nogc = df[(df["has_group_covariate"] == False) & (df["snr_level"] == "medium")]
    if gc.empty or nogc.empty:
        print("Insufficient group covariate data — skipping.")
        return

    for effect in df["effect_type"].unique():
        rows = []
        for label, sub in [("With group", gc[gc["effect_type"] == effect]),
                            ("Without group", nogc[nogc["effect_type"] == effect])]:
            for method in CLUSTERING_METHODS:
                col = f"{method}_Reconstruction_MSE"
                if col in sub.columns:
                    for v in sub[col].dropna():
                        rows.append({"Condition": label, "Method": METHOD_LABELS.get(method, method), "MSE": v})
        if not rows:
            continue
        plot_df = pd.DataFrame(rows)
        plt.figure(figsize=(10, 5))
        palette = {METHOD_LABELS.get(m, m): METHOD_COLORS.get(m, "gray") for m in CLUSTERING_METHODS}
        sns.barplot(data=plot_df, x="Method", y="MSE", hue="Condition",
                    palette=["#d62728", "#aec7e8"], errorbar="se")
        plt.title(f"Reconstruction MSE: With vs. Without Group Covariate — {effect}", fontweight="bold")
        plt.ylabel("Reconstruction MSE (log scale)")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"group_covariate_{effect}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")


# ── Table: Summary mean ± SD ──────────────────────────────────────────────────

def write_summary_table(df: pd.DataFrame, out_dir: str):
    """Write mean ± SD for key metrics per (condition_label, effect_type, method)."""
    metric_cols = (
        [(f"{m}_BestJaccard", m, "BestJaccard") for m in CLUSTERING_METHODS] +
        [(f"{m}_BestPrecision", m, "BestPrecision") for m in CLUSTERING_METHODS] +
        [(f"{m}_UnannotatedRecall", m, "UnannotatedRecall") for m in CLUSTERING_METHODS] +
        [(f"{m}_Reconstruction_MSE", m, "MSE") for m in CLUSTERING_METHODS + PATHWAY_METHODS] +
        [(f"{m}_Sensitivity", m, "Sensitivity") for m in PATHWAY_METHODS] +
        [(f"{m}_FPR", m, "FPR") for m in PATHWAY_METHODS]
    )

    rows = []
    group_cols = [c for c in ["condition_label", "effect_type"] if c in df.columns]
    for group_vals, gdf in df.groupby(group_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        for col, method, metric in metric_cols:
            if col not in gdf.columns:
                continue
            vals = gdf[col].dropna()
            if vals.empty:
                continue
            rows.append({
                **dict(zip(group_cols, group_vals)),
                "method": METHOD_LABELS.get(method, method),
                "metric": metric,
                "mean": round(vals.mean(), 4),
                "sd": round(vals.std(), 4),
                "n": len(vals),
            })

    if not rows:
        print("No data for summary table.")
        return
    summary = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "summary_table.csv")
    summary.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results across conditions.")
    parser.add_argument("base_dir", help="Base output directory containing benchmark_results.csv files")
    parser.add_argument("--out_dir", default=None, help="Output directory for figures (default: base_dir/figures)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.base_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading results from: {args.base_dir}")
    df = load_all(args.base_dir)
    print(f"Loaded {len(df)} total replicates from {df['source_path'].nunique()} files.")
    print(f"Conditions: {sorted(df['condition_label'].unique())}")
    print()

    sns.set_theme(style="whitegrid", font_scale=1.1)

    plot_annotation_sweep(df, out_dir)
    plot_snr_sweep(df, out_dir)
    plot_group_covariate(df, out_dir)
    write_summary_table(df, out_dir)

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
