"""
Main entry point for the longitudinal metabolomics simulation framework.
"""

import argparse
import os
import time
import concurrent.futures
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, Tuple

from .utils import (
    setup_environment, suppress_output, evaluate_module, 
    evaluate_clustering_performance, calculate_reconstruction_fidelity
)
from .effects import create_spike_effect, create_linear_increase_effect, create_periodic_effect
from .core import simulate_longitudinal_data
from .methods import (
    analyze_with_mogp, analyze_with_wgcna, analyze_with_mefisto,
    analyze_with_dpgp, analyze_with_timeomics, analyze_with_meba,
    analyze_with_pal, fit_metabolite_lmms, analyze_with_lmm_ora, analyze_with_lmm_gsea,
    analyze_with_mogp_gsea,
)
from .plots import visualize_benchmark_results

def run_single_replicate(run_id: int, seed: int, args) -> Tuple[Dict[str, Any], pd.DataFrame]:
    # Re-run environment setup in each worker process. ProcessPoolExecutor spawns
    # fresh processes that do not inherit os.environ changes made in the parent, so
    # TF_CPP_MIN_LOG_LEVEL, R_HOME, and rpy2 logging must be configured here too.
    setup_environment()
    np.random.seed(seed)
    verbose = bool(getattr(args, "debug", False)) or (args.n_runs == 1)

    # 1. Setup Effects
    if args.effect_type == "spike":
        effect_func = create_spike_effect(8, 12, args.effect_magnitude)
    else:
        duration = 20 - 6
        slope = args.effect_magnitude / max(1e-6, duration)
        effect_func = create_linear_increase_effect(slope, 6)

    nuisance_func = None
    if args.nuisance_fraction > 0:
        nuisance_func = create_periodic_effect(args.nuisance_amplitude, args.nuisance_period)

    # 2. Simulate Data
    df, true_pathways, annotated_pathways = simulate_longitudinal_data(
        n_subjects=args.n_subjects, n_metabolites=args.n_metabolites,
        n_pathways=args.n_pathways, metabolites_per_pathway=args.metabolites_per_pathway,
        time_points=np.linspace(0, 20, args.n_time_points),
        add_group_covariate=args.add_group_covariate,
        irregular_sampling_sd=args.irregular_sampling_sd,
        subject_random_effect_sd=args.subject_noise,
        pathway_random_effect_sd=args.pathway_noise,
        dispersion=args.dispersion,
        effect_func=effect_func,
        nuisance_fraction=args.nuisance_fraction,
        nuisance_effect_func=nuisance_func,
        annotation_fraction=args.annotation_fraction,
    )

    results = {"run_id": run_id, "seed": seed, "effect_type": args.effect_type,
               "condition_label": getattr(args, "condition_label", "")}
    all_fits = []
    active_pid = list(true_pathways.keys())[0]
    true_mets = set(true_pathways[active_pid])
    unannotated = true_mets - set(annotated_pathways[active_pid])
    all_ids = sorted(df["metabolite_id"].unique())

    def process(name, modules, fit, is_pathway=False):
        if not is_pathway:
            results[f"{name}_ARI"] = evaluate_clustering_performance(true_pathways, modules, all_ids, name, verbose=verbose)
            best_j, best_rec, best_prec = 0.0, 0.0, 0.0
            for _, mets in modules:
                m = evaluate_module(mets, true_mets, unannotated, verbose=False)
                if m["jaccard"] > best_j:
                    best_j, best_rec, best_prec = m["jaccard"], m["unannotated_recall"], m["precision"]
            results[f"{name}_BestJaccard"] = best_j
            results[f"{name}_BestPrecision"] = best_prec
            results[f"{name}_UnannotatedRecall"] = best_rec
            results[f"{name}_NumModules"] = len(modules)
        else:
            results[f"{name}_Sensitivity"] = 1 if active_pid in modules else 0
            inactive = set(true_pathways.keys()) - {active_pid}
            results[f"{name}_FPR"] = len(set(modules).intersection(inactive)) / len(inactive) if inactive else 0

        if not fit.empty:
            merged = pd.merge(fit, df, on=["subject_id", "time", "metabolite_id"], how="left")
            results[f"{name}_Reconstruction_MSE"] = calculate_reconstruction_fidelity(merged, true_mets)
            merged["method"] = name
            all_fits.append(merged)
        else:
            # Baseline: predict per-metabolite mean when no modules found
            baseline = df[df["metabolite_id"].isin(true_mets)].copy()
            baseline["fitted_value"] = baseline.groupby("metabolite_id")["true_mu"].transform("mean")
            results[f"{name}_Reconstruction_MSE"] = calculate_reconstruction_fidelity(baseline, true_mets)

    # 3. Run Methods
    methods = [
        ("MOGP", analyze_with_mogp, args.skip_mogp, False),
        ("WGCNA", analyze_with_wgcna, args.skip_wgcna, False),
        ("MEFISTO", analyze_with_mefisto, args.skip_mefisto, False),
        ("DPGP", analyze_with_dpgp, args.skip_dpgp, False),
    ]

    mogp_fit, mogp_W_df = pd.DataFrame(), pd.DataFrame()
    for name, func, skip, is_path in methods:
        if not skip:
            t0 = time.time()
            if name == "MOGP":
                mods, fit, mogp_W_df = func(df, verbose=verbose)
                mogp_fit = fit
            else:
                mods, fit = func(df, verbose=verbose)
            results[f"{name}_Time"] = time.time() - t0
            process(name, mods, fit, is_path)

    # MOGP_GSEA: preranked GSEA per latent factor using absolute loading weights |W_{ik}|.
    # Min-p + Bonferroni correction across factors; self-calibrating when horseshoe
    # prior concentrates signal onto K≈1 factor per pathway.
    if not args.skip_mogp and not mogp_W_df.empty:
        t0 = time.time()
        mogp_gsea_paths = analyze_with_mogp_gsea(mogp_W_df, annotated_pathways)
        # Total pipeline time = MOGP training + GSEA enrichment step
        results["MOGP_GSEA_Time"] = results.get("MOGP_Time", 0.0) + (time.time() - t0)
        process("MOGP_GSEA", mogp_gsea_paths, mogp_fit, is_pathway=True)

    if not args.skip_meba:
        t0 = time.time()
        paths, fit = analyze_with_meba(df, annotated_pathways, verbose=verbose)
        results["MEBA_Time"] = time.time() - t0
        process("MEBA", paths, fit, True)

    if not args.skip_pal and args.add_group_covariate:
        t0 = time.time()
        paths, fit = analyze_with_pal(df, annotated_pathways, verbose=verbose)
        results["PAL_Time"] = time.time() - t0
        process("PAL", paths, fit, True)

    # timeOmics (uses mixOmics::spls via BLAS/Accelerate) runs after all other
    # R-based methods (MEBA, PAL) to avoid corrupting their threading state on macOS.
    if not args.skip_timeomics:
        t0 = time.time()
        mods, fit = analyze_with_timeomics(df, verbose=verbose)
        results["timeOmics_Time"] = time.time() - t0
        process("timeOmics", mods, fit, False)

    if not args.skip_lmm:
        t0 = time.time()
        lmm_res, lmm_fit = fit_metabolite_lmms(df, verbose=verbose)
        lmm_fit_time = time.time() - t0
        results["LMM_Fit_Time"] = lmm_fit_time

        t0 = time.time()
        process("LMM_ORA", analyze_with_lmm_ora(lmm_res, annotated_pathways), lmm_fit, True)
        # Total pipeline time = LMM fitting + ORA enrichment step
        results["LMM_ORA_Time"] = lmm_fit_time + (time.time() - t0)

        t0 = time.time()
        process("LMM_GSEA", analyze_with_lmm_gsea(lmm_res, annotated_pathways), lmm_fit, True)
        # Total pipeline time = LMM fitting + GSEA enrichment step
        results["LMM_GSEA_Time"] = lmm_fit_time + (time.time() - t0)

    if getattr(args, "skip_fitted_predictions", False):
        return results, pd.DataFrame()
    final_fit = pd.concat(all_fits, ignore_index=True) if all_fits else pd.DataFrame()
    if not final_fit.empty: final_fit["run_id"] = run_id
    return results, final_fit

def main():
    setup_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_subjects", type=int, default=100)
    parser.add_argument("--n_metabolites", type=int, default=500)
    parser.add_argument("--n_pathways", type=int, default=5)
    parser.add_argument("--metabolites_per_pathway", type=int, default=20)
    parser.add_argument("--n_time_points", type=int, default=5)
    parser.add_argument("--annotation_fraction", type=float, default=0.7)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="simulation_results")
    parser.add_argument("--seed", type=int, default=9102)
    parser.add_argument("--skip_mogp", action="store_true")
    parser.add_argument("--skip_wgcna", action="store_true")
    parser.add_argument("--skip_lmm", action="store_true")
    parser.add_argument("--skip_mefisto", action="store_true")
    parser.add_argument("--skip_dpgp", action="store_true")
    parser.add_argument("--skip_timeomics", action="store_true")
    parser.add_argument("--skip_pal", action="store_true")
    parser.add_argument("--skip_meba", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip_fitted_predictions", action="store_true",
                        help="Skip writing fitted_predictions.csv (metrics in benchmark_results.csv are unaffected)")
    parser.add_argument("--condition_label", type=str, default="",
                        help="Label written to benchmark_results.csv to identify this condition in aggregate analysis")
    parser.add_argument("--add_group_covariate", action="store_true")
    parser.add_argument("--effect_type", type=str, default="spike", choices=["spike", "linear"])
    parser.add_argument("--effect_magnitude", type=float, default=2.5)
    parser.add_argument("--subject_noise", type=float, default=0.2)
    parser.add_argument("--pathway_noise", type=float, default=0.3)
    parser.add_argument("--irregular_sampling_sd", type=float, default=1.5)
    parser.add_argument("--dispersion", type=float, default=20.0)
    parser.add_argument("--nuisance_fraction", type=float, default=0.2)
    parser.add_argument("--nuisance_amplitude", type=float, default=1.5)
    parser.add_argument("--nuisance_period", type=float, default=10.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [args.seed + i for i in range(args.n_runs)]
    all_res, all_fit = [], []

    if args.n_jobs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            futures = {ex.submit(run_single_replicate, i + 1, s, args): i + 1 for i, s in enumerate(seeds)}
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(seeds)):
                run_id = futures[f]
                try:
                    r, fit = f.result()
                    all_res.append(r); all_fit.append(fit)
                except Exception as e:
                    print(f"\n[run {run_id}] FAILED: {type(e).__name__}: {e}", flush=True)
    else:
        for i, s in enumerate(seeds):
            try:
                r, fit = run_single_replicate(i + 1, s, args)
                all_res.append(r); all_fit.append(fit)
            except Exception as e:
                print(f"\n[run {i + 1}] FAILED: {type(e).__name__}: {e}", flush=True)

    if all_res:
        results_path = os.path.join(args.output_dir, "benchmark_results.csv")
        pd.DataFrame(all_res).to_csv(results_path, index=False)
        if not args.skip_fitted_predictions:
            fit_frames = [f for f in all_fit if isinstance(f, pd.DataFrame) and not f.empty]
            fit_out = pd.concat(fit_frames) if fit_frames else pd.DataFrame()
            fit_out.to_csv(os.path.join(args.output_dir, "fitted_predictions.csv"), index=False)

        # Automatically generate visualizations
        print(f"Generating visualizations in {args.output_dir}...")
        visualize_benchmark_results(results_path, args.output_dir)

if __name__ == "__main__":
    main()
