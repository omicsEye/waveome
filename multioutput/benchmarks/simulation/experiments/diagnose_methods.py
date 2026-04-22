#!/usr/bin/env python3
"""
diagnose_methods.py — Single-replicate diagnostic for MEFISTO, timeOmics, and LMM+GSEA.

Run from project root:
    /Users/allen/miniforge3/envs/mogp-waveome-sim/bin/python3.11 \
        code/simulation/experiments/diagnose_methods.py
"""

import os, sys
os.environ["R_HOME"] = "/Users/allen/miniforge3/envs/mogp-waveome-sim/lib/R"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from code.simulation.core import simulate_longitudinal_data
from code.simulation.effects import create_spike_effect, create_linear_increase_effect

np.random.seed(9102)

print("=" * 60)
print("Simulating easy-SNR linear dataset...")
print("=" * 60)

effect_func = create_linear_increase_effect(slope=4.0 / 14, intercept_time=6.0)

df, true_pathways, annotated_pathways = simulate_longitudinal_data(
    n_subjects=20, n_metabolites=100, n_pathways=5, metabolites_per_pathway=20,
    time_points=np.linspace(0, 20, 10),
    add_group_covariate=False, irregular_sampling_sd=0.5,
    subject_random_effect_sd=0.1, pathway_random_effect_sd=0.3,
    dispersion=50.0, effect_func=effect_func,
    nuisance_fraction=0.0, annotation_fraction=0.7,
)
active_pid = list(true_pathways.keys())[0]
true_mets = set(true_pathways[active_pid])
print(f"Active pathway: {active_pid}, {len(true_mets)} true metabolites")
print(f"Annotated: {len(annotated_pathways[active_pid])} metabolites")
print()

# ── LMM+GSEA DIAGNOSTIC ───────────────────────────────────────────────────────

print("=" * 60)
print("DIAGNOSTIC 1: LMM+GSEA vs LMM+ORA")
print("=" * 60)

from code.simulation.methods import fit_metabolite_lmms, analyze_with_lmm_ora, analyze_with_lmm_gsea

lmm_results, lmm_fit = fit_metabolite_lmms(df, verbose=False)

ora_result = analyze_with_lmm_ora(lmm_results, annotated_pathways)
print(f"LMM+ORA detected pathways: {ora_result}")
sig_count = (lmm_results["pvalue"] < 0.05).sum()
print(f"  Significant metabolites (p<0.05): {sig_count} / {len(lmm_results)}")
sig_mets = set(lmm_results[lmm_results["pvalue"] < 0.05]["metabolite_id"])
overlap = len(sig_mets & true_mets)
print(f"  Overlap with true active pathway mets: {overlap} / {len(true_mets)}")

print()
print("Running LMM+GSEA with detailed output...")
try:
    import gseapy as gp
    rnk = lmm_results.sort_values("tstat", ascending=False)[["metabolite_id", "tstat"]]
    print(f"  Top 5 ranked metabolites:")
    for _, row in rnk.head(5).iterrows():
        in_true = row["metabolite_id"] in true_mets
        print(f"    {row['metabolite_id']:20s}  tstat={row['tstat']:7.3f}  in_true={in_true}")
    print(f"  Bottom 5 ranked metabolites:")
    for _, row in rnk.tail(5).iterrows():
        in_true = row["metabolite_id"] in true_mets
        print(f"    {row['metabolite_id']:20s}  tstat={row['tstat']:7.3f}  in_true={in_true}")

    res = gp.prerank(rnk=rnk, gene_sets=annotated_pathways, threads=1,
                     outdir=None, seed=9102, verbose=False, min_size=5)
    print()
    print("  GSEA results table:")
    fdr_col = "FDR q-val" if "FDR q-val" in res.res2d.columns else "fdr"
    nom_col = "NOM p-val" if "NOM p-val" in res.res2d.columns else "pval"
    for _, row in res.res2d.iterrows():
        pathway = row.get("Term", row.name)
        is_active = (pathway == active_pid)
        print(f"    {pathway:20s}  NOM={row.get(nom_col, '?'):.4f}  FDR={row.get(fdr_col, '?'):.4f}  ES={row.get('ES', '?'):.3f}  active={is_active}")
    gsea_result = analyze_with_lmm_gsea(lmm_results, annotated_pathways)
    print(f"\n  LMM+GSEA detected pathways (FDR<0.05): {gsea_result}")
except Exception as e:
    print(f"  GSEA diagnostic failed: {e}")

print()

# ── MEFISTO DIAGNOSTIC ────────────────────────────────────────────────────────

print("=" * 60)
print("DIAGNOSTIC 2: MEFISTO factor variance ratios")
print("=" * 60)

try:
    import anndata as ad
    import muon as mu
    import tempfile

    data_copy = df.copy()
    data_copy["value"] = np.log1p(data_copy["value"])

    def get_standardized_sample_id(subject_id, time):
        return f"{subject_id}_{time:.4f}"

    sample_meta = data_copy[["subject_id", "time"]].drop_duplicates().reset_index(drop=True)
    sample_meta["sample_id"] = sample_meta.apply(
        lambda r: get_standardized_sample_id(r["subject_id"], r["time"]), axis=1)
    data_copy = pd.merge(data_copy, sample_meta, on=["subject_id", "time"])
    wide_data = data_copy.pivot_table(
        index="sample_id", columns="metabolite_id", values="value", aggfunc="mean").fillna(0)
    obs_meta = sample_meta.set_index("sample_id").loc[wide_data.index]

    print(f"  wide_data shape: {wide_data.shape}  (samples × metabolites)")
    print(f"  total_var of data: {np.var(wide_data.values):.6f}")

    adata = ad.AnnData(X=wide_data.values, obs=obs_meta)
    mdata = mu.MuData({"metabolomics": adata})
    mdata.obs["time"] = obs_meta.loc[mdata.obs_names, "time"].values
    mdata.obs["subject_id"] = obs_meta.loc[mdata.obs_names, "subject_id"].values

    tmp_h5 = tempfile.mktemp(suffix=".hdf5", prefix="mefisto_diag_")
    print("  Running MOFA/MEFISTO (this may take ~10s)...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu.tl.mofa(mdata, n_factors=min(10, wide_data.shape[1] - 1),
                   groups_label="subject_id", smooth_covariate="time",
                   smooth_kwargs={"scale_cov": True}, use_var=None,
                   quiet=True, outfile=tmp_h5)
    try:
        os.unlink(tmp_h5)
    except OSError:
        pass

    factors = np.asarray(mdata.obsm["X_mofa"])
    if "LFs" in mdata.mod["metabolomics"].varm:
        weights = np.asarray(mdata.mod["metabolomics"].varm["LFs"])
    else:
        weights = np.asarray(mdata.varm["LFs"])

    print(f"  factors shape: {factors.shape}  weights shape: {weights.shape}")
    total_var = np.var(wide_data.values)
    n_factors = weights.shape[1]
    print(f"  variance_threshold used in wrapper: 0.001")
    print()
    print(f"  {'Factor':>8}  {'var_ratio':>12}  {'passes':>8}  {'n_active_mets':>14}")
    for k in range(n_factors):
        outer = np.outer(factors[:, k], weights[:, k])
        var_ratio = np.var(outer) / total_var if total_var > 0 else 0.0
        max_w = np.max(np.abs(weights[:, k]))
        active = np.where((np.abs(weights[:, k]) / max_w) > 0.5)[0]
        passes = var_ratio >= 0.001
        print(f"  {k+1:>8}  {var_ratio:>12.6f}  {str(passes):>8}  {len(active):>14}")

    print()
    # Show what modules would be found at different thresholds
    for thresh in [0.001, 0.0001, 0.00001]:
        found = sum(1 for k in range(n_factors)
                    if np.var(np.outer(factors[:, k], weights[:, k])) / total_var >= thresh)
        print(f"  variance_threshold={thresh:.5f} → {found} factors pass")

except Exception as e:
    import traceback
    print(f"  MEFISTO diagnostic failed: {e}")
    traceback.print_exc()

print()

# ── timeOmics DIAGNOSTIC ──────────────────────────────────────────────────────

print("=" * 60)
print("DIAGNOSTIC 3: timeOmics metabolite ID mapping")
print("=" * 60)

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    data_log = df.copy()
    data_log["value"] = np.log1p(data_log["value"])
    data_log = data_log.sort_values(["metabolite_id", "subject_id", "time"])

    keepX_val = max(1, int(df["metabolite_id"].nunique() / 5))
    print(f"  keepX_val = {keepX_val}")
    print(f"  Python-side true_mets sample: {sorted(list(true_mets))[:5]}")
    print(f"  Python-side metabolite_id sample: {sorted(df['metabolite_id'].unique())[:5]}")

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["df_long"] = ro.conversion.py2rpy(data_log)
    ro.globalenv["keepX_val"] = keepX_val

    ro.r("""
        library(timeOmics); library(mixOmics); library(lmms)
        run_timeomics_diag <- function(df, keepX_val) {
            unique_mets <- unique(df$metabolite_id)
            n_mets <- length(unique_mets)
            n_obs <- nrow(df) / n_mets
            cat(sprintf("  R: n_mets=%d  n_obs=%.1f  n_obs%%1=%.6f\\n", n_mets, n_obs, n_obs %% 1))
            cat(sprintf("  R: first 5 unique_mets: %s\\n",
                paste(head(unique_mets, 5), collapse=", ")))
            if (n_obs %% 1 != 0) { cat("  R: n_obs not integer — returning NULL\\n"); return(NULL) }

            mat <- matrix(df$value, nrow=n_obs, ncol=n_mets, byrow=FALSE)
            colnames(mat) <- unique_mets
            time_vec <- as.numeric(df$time[1:n_obs])
            sampleID_vec <- as.character(df$subject_id[1:n_obs])
            cat(sprintf("  R: mat dim: %d x %d\\n", nrow(mat), ncol(mat)))
            cat(sprintf("  R: time_vec[1:5]: %s\\n", paste(round(head(time_vec, 5), 3), collapse=", ")))

            grid_points <- 10
            pred_time <- seq(min(time_vec), max(time_vec), length.out=grid_points)
            models <- tryCatch({
                utils::capture.output({
                    res <- lmms::lmmSpline(data=mat, time=time_vec, sampleID=sampleID_vec,
                                           timePredict=pred_time, keepModels=TRUE, numCores=1)
                })
                res
            }, error = function(e) { cat(sprintf("  R: lmmSpline error: %s\\n", e$message)); NULL })
            if (is.null(models)) return(NULL)

            n_splines <- length(models@predSpline[[1]])
            spline_names <- head(names(models@predSpline[[1]]), 5)
            cat(sprintf("  R: lmmSpline fitted %d metabolite splines\\n", n_splines))
            cat(sprintf("  R: first 5 spline names: %s\\n", paste(spline_names, collapse=", ")))

            X_mat <- do.call(rbind, lapply(models@predSpline[[1]], as.numeric))
            X_mat[is.na(X_mat)] <- 0
            X <- t(X_mat)
            cat(sprintf("  R: X matrix dim (grid_pts x mets): %d x %d\\n", nrow(X), ncol(X)))
            cat(sprintf("  R: first 5 colnames(X): %s\\n", paste(head(colnames(X), 5), collapse=", ")))

            Y <- matrix(1:nrow(X), ncol=1)
            res_spls <- tryCatch({
                mixOmics::spls(X=X, Y=Y, ncomp=1, keepX=keepX_val, mode='regression', scale=TRUE)
            }, error = function(e) { cat(sprintf("  R: spls error: %s\\n", e$message)); NULL })
            if (is.null(res_spls)) return(NULL)

            loadings <- res_spls$loadings$X[,1]
            active <- names(loadings)[abs(loadings) > 1e-10]
            cat(sprintf("  R: spls selected %d metabolites\\n", length(active)))
            cat(sprintf("  R: first 5 selected: %s\\n", paste(head(active, 5), collapse=", ")))

            clusters <- tryCatch({ timeOmics::getCluster(res_spls) }, error = function(e) NULL)
            if (is.null(clusters) || nrow(clusters) == 0) {
                cat("  R: getCluster returned NULL/empty — using loading fallback\\n")
                cluster_ids <- ifelse(loadings[active] >= 0, "1", "-1")
                all_ids <- setNames(rep("grey", length(loadings)), names(loadings))
                all_ids[active] <- cluster_ids
                n1 <- sum(cluster_ids == "1"); nm1 <- sum(cluster_ids == "-1")
                cat(sprintf("  R: fallback clusters: %d positive, %d negative, %d grey\\n",
                            n1, nm1, length(loadings) - length(active)))
                return(all_ids)
            }
            cat(sprintf("  R: getCluster found %d rows\\n", nrow(clusters)))
            cat(sprintf("  R: cluster$molecule sample: %s\\n",
                paste(head(clusters$molecule, 5), collapse=", ")))
            setNames(as.character(clusters$cluster), clusters$molecule)
        }
    """)

    r_clusters = ro.r("run_timeomics_diag(df_long, keepX_val)")
    if r_clusters is None or not hasattr(r_clusters, 'names'):
        print("  Python: r_clusters is None or has no names")
    else:
        cluster_map = dict(zip(r_clusters.names, list(r_clusters)))
        modules = {}
        for m, c in cluster_map.items():
            modules.setdefault(f"timeOmics_Cluster_{c}", []).append(str(m))

        print(f"\n  Python cluster_map sample (first 10):")
        for m, c in list(cluster_map.items())[:10]:
            in_true = m in true_mets
            print(f"    {m:20s} → cluster={c:5s}  in_true_mets={in_true}")

        print(f"\n  Module sizes:")
        for mod_name, mets in modules.items():
            overlap = len(set(mets) & true_mets)
            print(f"    {mod_name}: {len(mets)} metabolites, overlap with true_mets={overlap}")

        print(f"\n  Python true_mets: {sorted(list(true_mets))[:5]}...")
        print(f"  Python cluster keys sample: {list(cluster_map.keys())[:5]}")
        print(f"  Type check: cluster key type={type(list(cluster_map.keys())[0])}, "
              f"true_met type={type(list(true_mets)[0])}")

except Exception as e:
    import traceback
    print(f"  timeOmics diagnostic failed: {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("Diagnostic complete.")
print("=" * 60)
