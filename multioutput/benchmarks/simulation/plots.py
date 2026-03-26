
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_fitted_prediction_plot(data_file: str, output_dir: str):
    """
    Creates a plot showing the true vs. predicted latent signals.
    """
    if not os.path.exists(data_file):
        print(f"Info: Fitted prediction data file not found at {data_file}. Skipping plot.")
        return

    if os.path.getsize(data_file) == 0:
        print("Info: Fitted prediction file is empty. Skipping plot.")
        return

    try:
        df = pd.read_csv(data_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        print("Info: Fitted prediction file is empty or malformed. Skipping plot.")
        return

    if df.empty or 'metabolite_id' not in df.columns:
        print("Info: Fitted prediction file has no usable rows. Skipping plot.")
        return
    
    # Check if 'is_active' exists
    if 'is_active' not in df.columns:
        print("Warning: 'is_active' column missing in fitted predictions. Using first metabolite.")
        metabolite_to_plot = df['metabolite_id'].unique()[0]
    else:
        active_mets = df[df['is_active'] == 1]['metabolite_id'].unique()
        if len(active_mets) > 0:
            metabolite_to_plot = active_mets[0]
        else:
            print("Warning: No active metabolites found. Using first metabolite.")
            metabolite_to_plot = df['metabolite_id'].unique()[0]

    plot_df = df[df['metabolite_id'] == metabolite_to_plot].copy()

    # Identify methods
    if 'method' in df.columns:
        methods = df['method'].unique()
    else:
        # Fallback for old wide format
        methods = [c.replace('pred_', '') for c in df.columns if c.startswith('pred_')]
    
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 1, figsize=(10, 6 * n_methods), sharex=True, sharey=True)
    if n_methods == 1:
        axes = [axes]

    for i, method in enumerate(methods):
        ax = axes[i]
        
        # Filter for method if long format
        if 'method' in df.columns:
            method_df = plot_df[plot_df['method'] == method]
            # Use 'fitted_value' column
            pred_col = 'fitted_value'
        else:
            method_df = plot_df
            pred_col = f'pred_{method}'
        
        # Plot raw data, colored by group
        if 'group' in method_df.columns:
            sns.scatterplot(data=method_df, x='time', y='value', hue='group', ax=ax, alpha=0.6, s=50, palette='viridis')
        else:
            sns.scatterplot(data=method_df, x='time', y='value', ax=ax, alpha=0.6, s=50)
            
        # Plot true latent signal if available
        if 'true_mu' in method_df.columns:
            # Use log of the values since the models predict on the log scale (usually)
            # Check if true_mu is already log or not? 
            # In simulation_framework, true_mu is exp(latent_signal). 
            # We plot log1p(true_mu) to match log-scale predictions if predictions are log-scale?
            # Wait, predictions in simulation_framework are returned as 'fitted_value' which is usually on the original scale (expm1 applied).
            # But the plot label says "Log(Value)".
            # Let's plot everything on Log(Value) scale.
            sns.lineplot(data=method_df, x='time', y=np.log1p(method_df['true_mu']), ax=ax, color='black', label='True Latent Signal', linestyle='--', linewidth=2.5)
        
        # Plot Predicted Signal
        # Assuming fitted_value is on original scale
        sns.lineplot(data=method_df, x='time', y=np.log1p(method_df[pred_col]), ax=ax, color='red', label=f'{method} Predicted Signal', linewidth=2.5)

        ax.set_title(f"Fitted Prediction vs. True Signal for {method}")
        ax.set_ylabel("Log(Value)")
        ax.legend()

    plt.xlabel("Time")
    fig.suptitle(f"Model Fits for Metabolite {metabolite_to_plot}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "fitted_prediction_comparison.png"), dpi=300)
    plt.close()
    print(f"Saved fitted_prediction_comparison.png")


def create_reconstruction_fidelity_plot(df: pd.DataFrame, output_dir: str):
    """
    Creates a boxplot comparing the Reconstruction MSE across different methods.
    """
    mse_cols = [c for c in df.columns if c.endswith("_Reconstruction_MSE")]
    if not mse_cols:
        print("Info: No Reconstruction MSE columns found. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    df_mse = df.melt(
        id_vars=["run_id"], 
        value_vars=mse_cols, 
        var_name="Method", 
        value_name="Reconstruction MSE"
    )
    df_mse["Method"] = df_mse["Method"].str.replace("_Reconstruction_MSE", "")
    
    if not df_mse.empty:
        sns.boxplot(
            data=df_mse, 
            x="Method", 
            y="Reconstruction MSE", 
            hue="Method", 
            palette="plasma", 
            legend=False
        )
        sns.stripplot(
            data=df_mse, 
            x="Method", 
            y="Reconstruction MSE", 
            color="black", 
            alpha=0.5, 
            jitter=True
        )
        
        plt.title("Active Pathway Signal Reconstruction Fidelity (MSE)")
        plt.ylabel("Reconstruction MSE (log1p scale)")
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Reconstruction_MSE_comparison.png"), dpi=300)
        plt.close()
        print(f"Saved Reconstruction_MSE_comparison.png")



def visualize_benchmark_results(results_file: str, output_dir: str = None):
    """
    Visualizes the results from the simulation framework benchmark.
    """
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return

    # Load data
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} runs from {results_file}")

    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Run all plotting functions ---
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # --- 1. ARI Comparison (Global Structure) ---
    ari_cols = [c for c in df.columns if c.endswith("_ARI")]
    if ari_cols:
        plt.figure(figsize=(10, 6))
        df_ari = df.melt(id_vars=["run_id"], value_vars=ari_cols, var_name="Method", value_name="ARI")
        df_ari["Method"] = df_ari["Method"].str.replace("_ARI", "")
        
        # Filter out LMM as it is a univariate method
        df_ari = df_ari[df_ari["Method"] != "LMM"]
        
        if not df_ari.empty:
            sns.boxplot(data=df_ari, x="Method", y="ARI", hue="Method", palette="viridis", legend=False)
            sns.stripplot(data=df_ari, x="Method", y="ARI", color="black", alpha=0.5, jitter=True)
            
            plt.title("Overall Clustering Performance (ARI)")
            plt.ylabel("Adjusted Rand Index")
            plt.ylim(-0.1, 1.1)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "ARI_comparison.png"), dpi=300)
            plt.close()
            print(f"Saved ARI_comparison.png")

    # --- 2. Best Jaccard Index (Active Pathway Detection) ---
    jaccard_cols = [c for c in df.columns if c.endswith("_BestJaccard")]
    if jaccard_cols:
        plt.figure(figsize=(10, 6))
        df_jaccard = df.melt(id_vars=["run_id"], value_vars=jaccard_cols, var_name="Method", value_name="Jaccard")
        df_jaccard["Method"] = df_jaccard["Method"].str.replace("_BestJaccard", "")
        
        # Filter out LMM
        df_jaccard = df_jaccard[df_jaccard["Method"] != "LMM"]
        
        if not df_jaccard.empty:
            sns.boxplot(data=df_jaccard, x="Method", y="Jaccard", hue="Method", palette="magma", legend=False)
            sns.stripplot(data=df_jaccard, x="Method", y="Jaccard", color="black", alpha=0.5, jitter=True)
            
            plt.title("Best Matching Module for Active Pathway (Jaccard Index)")
            plt.ylabel("Jaccard Index")
            plt.ylim(-0.1, 1.1)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "Jaccard_comparison.png"), dpi=300)
            plt.close()
            print(f"Saved Jaccard_comparison.png")

    # --- 3. Module Precision ---
    precision_cols = [c for c in df.columns if c.endswith("_BestPrecision")]
    if precision_cols:
        plt.figure(figsize=(10, 6))
        df_prec = df.melt(
            id_vars=["run_id"], value_vars=precision_cols,
            var_name="Method", value_name="Precision",
        )
        df_prec["Method"] = df_prec["Method"].str.replace("_BestPrecision", "")
        df_prec = df_prec[df_prec["Method"] != "LMM"]
        if not df_prec.empty:
            sns.boxplot(
                data=df_prec, x="Method", y="Precision",
                hue="Method", palette="flare", legend=False,
            )
            sns.stripplot(
                data=df_prec, x="Method", y="Precision",
                color="black", alpha=0.5, jitter=True,
            )
            plt.title("Module Precision for Best-Matching Module (Active Pathway)")
            plt.ylabel("Precision")
            plt.ylim(-0.1, 1.1)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "Precision_comparison.png"), dpi=300
            )
            plt.close()
            print("Saved Precision_comparison.png")

    # --- 4. Unannotated Metabolite Recall ---
    recall_cols = [c for c in df.columns if c.endswith("_UnannotatedRecall")]
    if recall_cols:
        plt.figure(figsize=(10, 6))
        df_recall = df.melt(id_vars=["run_id"], value_vars=recall_cols, var_name="Method", value_name="Recall")
        df_recall["Method"] = df_recall["Method"].str.replace("_UnannotatedRecall", "")
        
        # Filter out LMM
        df_recall = df_recall[df_recall["Method"] != "LMM"]
        
        if not df_recall.empty:
            sns.boxplot(data=df_recall, x="Method", y="Recall", hue="Method", palette="rocket", legend=False)
            sns.stripplot(data=df_recall, x="Method", y="Recall", color="black", alpha=0.5, jitter=True)
            
            plt.title("Recall of Unannotated Metabolites in Active Pathway")
            plt.ylabel("Recall")
            plt.ylim(-0.1, 1.1)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "Recall_comparison.png"), dpi=300)
            plt.close()
            print(f"Saved Recall_comparison.png")

    # --- 4. Number of Modules Comparison ---
    num_modules_cols = [c for c in df.columns if c.endswith("_NumModules")]
    if num_modules_cols:
        plt.figure(figsize=(10, 6))
        df_modules = df.melt(id_vars=["run_id"], value_vars=num_modules_cols, var_name="Method", value_name="Num Modules")
        df_modules["Method"] = df_modules["Method"].str.replace("_NumModules", "")
        
        # Filter out LMM
        df_modules = df_modules[df_modules["Method"] != "LMM"]
        
        if not df_modules.empty:
            sns.boxplot(data=df_modules, x="Method", y="Num Modules", hue="Method", palette="crest", legend=False)
            sns.stripplot(data=df_modules, x="Method", y="Num Modules", color="black", alpha=0.5, jitter=True)
            
            plt.title("Number of Modules Detected per Method")
            plt.ylabel("Count of Modules Found")
            plt.ylim(bottom=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "NumModules_comparison.png"), dpi=300)
            plt.close()
            print(f"Saved NumModules_comparison.png")

    # --- 5. Execution Time Comparison ---
    plot_rows = []
    
    # 1. Add non-LMM methods
    non_lmm_methods = [c.replace("_Time", "") for c in df.columns if c.endswith("_Time") and "LMM" not in c]
    for m in non_lmm_methods:
        for _, row in df.iterrows():
            plot_rows.append({"Method": m, "Time (s)": row[f"{m}_Time"], "run_id": row["run_id"]})
    
    # 2. Add LMM variants
    if "LMM_Fit_Time" in df.columns:
        # Check for ORA
        ora_col = [c for c in df.columns if "LMM_ORA" in c and c.endswith("_Time")]
        for _, row in df.iterrows():
            ora_extra = row[ora_col[0]] if ora_col else 0
            plot_rows.append({"Method": "LMM+ORA", "Time (s)": row["LMM_Fit_Time"] + ora_extra, "run_id": row["run_id"]})
            
        # Check for GSEA
        gsea_col = [c for c in df.columns if "LMM_GSEA" in c and c.endswith("_Time")]
        for _, row in df.iterrows():
            gsea_extra = row[gsea_col[0]] if gsea_col else 0
            plot_rows.append({"Method": "LMM+GSEA", "Time (s)": row["LMM_Fit_Time"] + gsea_extra, "run_id": row["run_id"]})

    df_time = pd.DataFrame(plot_rows)

    if not df_time.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_time, x="Method", y="Time (s)", hue="Method", errorbar="sd", palette="coolwarm", legend=False)
        
        plt.title("Average Execution Time per Method")
        plt.ylabel("Time (seconds)")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Time_comparison.png"), dpi=300)
        plt.close()
        print(f"Saved Time_comparison.png")

    # --- 6. Pathway-Aware Method Performance ---
    pathway_methods = ["LMM_ORA", "LMM_GSEA", "MEBA", "MOGP_ORA", "MOGP_GSEA", "PAL"]
    sensitivity_cols = [f"{m}_Sensitivity" for m in pathway_methods if f"{m}_Sensitivity" in df.columns]
    fpr_cols = [f"{m}_FPR" for m in pathway_methods if f"{m}_FPR" in df.columns]

    if sensitivity_cols and fpr_cols:
        sensitivity_means = df[sensitivity_cols].mean() * 100
        fpr_means = df[fpr_cols].mean() * 100
        clean_labels = [c.replace("_Sensitivity", "").replace("_", "+") for c in sensitivity_cols]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.barplot(x=clean_labels, y=sensitivity_means.values, hue=clean_labels, palette="Greens_d", ax=ax1, legend=False)
        ax1.set_title("Sensitivity (True Positive Rate)")
        ax1.set_ylabel("Sensitivity (%)")
        ax1.set_ylim(0, 105)
        for i, v in enumerate(sensitivity_means.values):
            ax1.text(i, v + 2, f"{v:.1f}%", ha='center')

        sns.barplot(x=clean_labels, y=fpr_means.values, hue=clean_labels, palette="Reds_d", ax=ax2, legend=False)
        ax2.set_title("False Positive Rate")
        ax2.set_ylabel("FPR (%)")
        ax2.set_ylim(0, 105)
        for i, v in enumerate(fpr_means.values):
            ax2.text(i, v + 2, f"{v:.1f}%", ha='center')
        
        fig.suptitle("Performance of Pathway-Aware Methods", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, "Pathway_method_performance.png"), dpi=300)
        plt.close()
        print(f"Saved Pathway_method_performance.png")

    # --- 7. Fitted Prediction Plot ---
    prediction_data_file = os.path.join(output_dir, "fitted_predictions.csv")
    create_fitted_prediction_plot(prediction_data_file, output_dir)

    # --- 8. Reconstruction Fidelity Plot ---
    create_reconstruction_fidelity_plot(df, output_dir)


if __name__ == "__main__":
    # Add numpy to scope for the new plotting function
    import numpy
    parser = argparse.ArgumentParser(description="Visualize simulation benchmark results.")
    parser.add_argument("results_file", type=str, help="Path to benchmark_results.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots")
    
    args = parser.parse_args()
    
    visualize_benchmark_results(args.results_file, args.output_dir)
