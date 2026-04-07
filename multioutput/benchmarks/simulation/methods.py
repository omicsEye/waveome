"""
Analysis methods for benchmarking against simulated longitudinal data.
"""

import os
import shutil
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_r_dependencies, get_standardized_sample_id, suppress_output

# =============================================================================
# 0. Internal Utilities
# =============================================================================


def _otsu_threshold(values: np.ndarray) -> float:
    """Otsu's method: find threshold minimizing within-class variance of |W| column."""
    vals = values.flatten()
    candidates = np.unique(vals)
    best_t, best_var = candidates[0], np.inf
    total_mean = vals.mean()
    n = len(vals)
    for t in candidates[:-1]:
        below = vals[vals <= t]
        above = vals[vals > t]
        if len(below) == 0 or len(above) == 0:
            continue
        w0, w1 = len(below) / n, len(above) / n
        within_var = w0 * below.var() + w1 * above.var()
        if within_var < best_var:
            best_var = within_var
            best_t = t
    return best_t


# =============================================================================
# 1. Analysis Utilities
# =============================================================================


def calculate_centroid_predictions(
    data: pd.DataFrame, modules: List[Tuple[str, List[str]]]
) -> pd.DataFrame:
    """Calculates centroid-based predictions for each metabolite based on module assignment."""
    if not modules: return pd.DataFrame()
    data_copy = data.copy()
    data_copy["log_value"] = np.log1p(data_copy["value"])
    data_copy["sample_id"] = data_copy.apply(lambda row: get_standardized_sample_id(row["subject_id"], row["time"]), axis=1)
    
    original_metabolites = sorted(data["metabolite_id"].unique())
    metab_to_mod = {}
    for mod_name, mets in modules:
        for m in mets:
            m_str = str(m).strip()
            if m_str.startswith("X") and m_str[1:].isdigit():
                try:
                    idx = int(m_str[1:]) - 1
                    if 0 <= idx < len(original_metabolites): m_str = original_metabolites[idx]
                except ValueError: pass
            metab_to_mod[m_str] = mod_name

    data_copy["module"] = data_copy["metabolite_id"].map(metab_to_mod)
    module_data = data_copy.dropna(subset=["module"])
    if module_data.empty: return pd.DataFrame()

    centroids = module_data.groupby(["sample_id", "module"])["log_value"].mean().reset_index()
    centroids.rename(columns={"log_value": "fitted_log"}, inplace=True)

    preds = data_copy.copy()
    preds["module"] = preds["metabolite_id"].map(metab_to_mod)
    preds = preds.merge(centroids, on=["sample_id", "module"], how="left")
    preds["fitted_log"] = preds["fitted_log"].fillna(0)
    preds["fitted_value"] = np.expm1(preds["fitted_log"])

    return preds[["subject_id", "time", "metabolite_id", "fitted_value"]]

# =============================================================================
# 2. Benchmarking Methods
# =============================================================================

def analyze_with_mogp(
    data: pd.DataFrame,
    weight_threshold: float = 0.2,
    use_otsu: bool = True,
    verbose: bool = True,
) -> Tuple[List[Tuple[str, List[str]]], pd.DataFrame]:
    if verbose: print("Running MOGP analysis...")
    try:
        import gpflow

        from waveome.model_search import GPSearch
    except ImportError:
        return [], pd.DataFrame()

    data_mod = data.copy()
    try:
        data_mod["subject_num"] = data_mod["subject_id"].astype(str).str.extract(r"(\d+)").astype(float)
    except Exception:
        data_mod["subject_num"] = data_mod["subject_id"].astype("category").cat.codes

    covariates = ["subject_num", "time"]
    if "group" in data.columns: covariates.append("group")

    wide_df = data_mod.pivot_table(index=covariates, columns="metabolite_id", values="value").reset_index()
    time_mean, time_std = wide_df["time"].mean(), wide_df["time"].std()
    if time_std > 0: wide_df["time"] = (wide_df["time"] - time_mean) / time_std

    metabolite_cols = [c for c in wide_df.columns if c not in covariates]
    X, Y = wide_df[covariates], wide_df[metabolite_cols]

    try:
        with suppress_output(not verbose):
            ms = GPSearch(X, Y, unit_col="subject_num", categorical_vars=["group"] if "group" in covariates else [], outcome_likelihood="negativebinomial")
            ms.multioutput_penalized_optimization(
                penalization_factor=1.0,
                num_opt_iter=50000,
                verbose=bool(verbose),
                random_seed=9102,
                adam_learning_rate=0.001,
            )
            ms.models["multioutput"].prune_latent_factors(
                loading_threshold=0.1, kl_threshold=0.01
            )
    except (Exception, SystemExit) as e:
        print(f"  MOGP failed: {e}")
        return [], pd.DataFrame()

    fitted_df_long = pd.DataFrame()
    if "multioutput" in ms.models:
        try:
            with suppress_output(not verbose):
                fitted_mu, _ = ms.models["multioutput"].predict_y(ms.X.to_numpy())
            fitted_df_wide = pd.DataFrame(np.array(fitted_mu), columns=metabolite_cols)
            full_wide = pd.concat([wide_df[covariates].reset_index(drop=True), fitted_df_wide], axis=1)
            if time_std > 0: full_wide["time"] = full_wide["time"] * time_std + time_mean
            fitted_df_long = full_wide.melt(id_vars=covariates, value_vars=metabolite_cols, var_name="metabolite_id", value_name="fitted_value")
            subject_map = data_mod[["subject_id", "subject_num"]].drop_duplicates()
            fitted_df_long = fitted_df_long.merge(subject_map, on="subject_num", how="left")[["subject_id", "time", "metabolite_id", "fitted_value"]]
        except (Exception, SystemExit) as e:
            print(f"  MOGP prediction failed: {e}")

    modules = []
    W_df = pd.DataFrame()
    if "multioutput" in ms.models:
        W = ms.models["multioutput"].kernel.W.numpy()
        factor_names = [f"MOGP_Factor_{k+1}" for k in range(W.shape[1])]
        W_df = pd.DataFrame(W, index=metabolite_cols, columns=factor_names)
        for k in range(W.shape[1]):
            abs_w = np.abs(W[:, k])
            max_w = abs_w.max()
            if max_w < 1e-6:
                continue
            # Use Otsu's method to find the natural signal/noise split in the
            # horseshoe-prior loading distribution. Falls back to relative threshold.
            if use_otsu and len(np.unique(abs_w)) > 1:
                try:
                    t = _otsu_threshold(abs_w)
                except Exception:
                    t = max_w * weight_threshold
            else:
                t = max_w * weight_threshold
            active = np.where(abs_w > t)[0]
            module_mets = [metabolite_cols[i] for i in active]
            if module_mets:
                modules.append((f"MOGP_Factor_{k+1}", module_mets))

    return modules, fitted_df_long, W_df

_wgcna_patched = False


def analyze_with_wgcna(data: pd.DataFrame, verbose: bool = True) -> Tuple[List[Tuple[str, List[str]]], pd.DataFrame]:
    if verbose: print("Running WGCNA analysis...")
    try:
        import PyWGCNA
        global _wgcna_patched
        if not _wgcna_patched:
            # PyWGCNA.WGCNA.cutreeHybrid() has an early-return path that yields a
            # DataFrame with column 'labels' instead of the expected 'Value' column.
            # labels2colors() always reads .Value, so we normalize the return here.
            _orig_cth = PyWGCNA.WGCNA.cutreeHybrid

            @staticmethod
            def _safe_cutreeHybrid(dendro, distM, **kwargs):
                result = _orig_cth(dendro, distM, **kwargs)
                if isinstance(result, pd.DataFrame) and "Value" not in result.columns:
                    col = result.iloc[:, 0]
                    result = pd.DataFrame(
                        {"Name": col.tolist(), "Value": col.tolist()}
                    )
                return result

            PyWGCNA.WGCNA.cutreeHybrid = _safe_cutreeHybrid
            _wgcna_patched = True

        data_copy = data.copy()
        data_copy["value"] = np.log1p(data_copy["value"])
        data_copy["sample_id"] = data_copy["subject_id"] + "_" + data_copy["time"].round(2).astype(str)
        wide_data = data_copy.pivot_table(index="sample_id", columns="metabolite_id", values="value", aggfunc="mean").fillna(0)
        with suppress_output(not verbose):
            pyWGCNA_obj = PyWGCNA.WGCNA(name="sim", species="homo sapiens", geneExp=wide_data, save=False, outputPath="", powers=list(range(1, 21)))
            pyWGCNA_obj.preprocess(show=False)
            pyWGCNA_obj.findModules(kwargs_function={"cutreeHybrid": {"deepSplit": 2}})
        module_assignments = pyWGCNA_obj.datExpr.var.get("dynamicColors")
        if module_assignments is not None:
            modules = [(name, list(mets)) for name, mets in module_assignments.groupby(module_assignments).groups.items() if name != "grey"]
            return modules, calculate_centroid_predictions(data, modules)
    except (Exception, SystemExit):
        # PyWGCNA calls sys.exit() when all genes land in one module.
        # dynamicColors is assigned before that call, so we can still
        # return the (degenerate) single-module result.
        try:
            module_assignments = pyWGCNA_obj.datExpr.var.get("dynamicColors")
            if module_assignments is not None:
                modules = [(name, list(mets)) for name, mets in module_assignments.groupby(module_assignments).groups.items() if name != "grey"]
                if modules:
                    return modules, calculate_centroid_predictions(data, modules)
        except Exception:
            pass
    return [], pd.DataFrame()


def analyze_with_mefisto(data: pd.DataFrame, weight_threshold: float = 0.5, variance_threshold: float = 0.001, verbose: bool = True) -> Tuple[List[Tuple[str, List[str]]], pd.DataFrame]:
    if verbose: print("Running MEFISTO analysis...")
    try:
        import anndata as ad
        import muon as mu
        data_copy = data.copy()
        data_copy["value"] = np.log1p(data_copy["value"])
        sample_meta = data_copy[["subject_id", "time"]].drop_duplicates().reset_index(drop=True)
        sample_meta["sample_id"] = sample_meta.apply(lambda r: get_standardized_sample_id(r["subject_id"], r["time"]), axis=1)
        data_copy = pd.merge(data_copy, sample_meta, on=["subject_id", "time"])
        wide_data = data_copy.pivot_table(index="sample_id", columns="metabolite_id", values="value", aggfunc="mean").fillna(0)
        obs_meta = sample_meta.set_index("sample_id").loc[wide_data.index]
        adata = ad.AnnData(X=wide_data.values, obs=obs_meta)
        mdata = mu.MuData({"metabolomics": adata})
        # muon >=0.1.5: smooth_covariate and groups_label look in mdata.obs
        mdata.obs["time"] = obs_meta.loc[mdata.obs_names, "time"].values
        mdata.obs["subject_id"] = obs_meta.loc[mdata.obs_names, "subject_id"].values
        import os as _os
        import tempfile
        tmp_h5 = tempfile.mktemp(suffix=".hdf5", prefix="mefisto_")
        try:
            with suppress_output(not verbose):
                mu.tl.mofa(
                    mdata,
                    n_factors=min(10, wide_data.shape[1] - 1),
                    groups_label="subject_id",
                    smooth_covariate="time",
                    smooth_kwargs={"scale_cov": True, "n_grid": 10},
                    use_var=None,
                    # Disable spike-and-slab/ARD weight priors: in single-view MOFA these
                    # priors are too aggressive and collapse all factor weights to zero
                    # before meaningful optimisation can occur. They were designed for
                    # multi-view settings where cross-view constraints stabilise sparsity.
                    spikeslab_weights=False,
                    ard_weights=False,
                    # Iteration budget: at N=subjects*timepoints=200, each ELBO iteration
                    # costs ~60s due to O(N^3) slogdet of the N×N variational posterior
                    # covariance (Z_nodes_GP_mv.calculateELBO_k, lb_q term). Full
                    # convergence (~200 iterations, ~2h) is impractical per replicate.
                    # 100 iterations (~100 min) allows more lengthscale optimisation while
                    # remaining feasible on HPC. Sparse GP does not reduce this bottleneck
                    # in mofapy2 v0.7.3 (posterior remains N×N). See manuscript.
                    n_iterations=100,
                    quiet=True,
                    outfile=tmp_h5,
                )
        finally:
            try:
                _os.unlink(tmp_h5)
            except OSError:
                pass
        
        # Factors are at MuData level; loadings may be at modality or MuData level
        # depending on muon version (confirmed by "Saved ... in .obsm['X_mofa']" message)
        factors = np.asarray(mdata.obsm["X_mofa"])
        if "LFs" in mdata.mod["metabolomics"].varm:
            weights = np.asarray(mdata.mod["metabolomics"].varm["LFs"])
        else:
            weights = np.asarray(mdata.varm["LFs"])
        total_var = np.var(wide_data.values)
        n_factors = weights.shape[1]
        modules = []
        for k in range(n_factors):
            var_ratio = np.var(np.outer(factors[:, k], weights[:, k])) / total_var if total_var > 0 else 0.0
            if var_ratio >= variance_threshold:
                max_w = np.max(np.abs(weights[:, k]))
                active = np.where((np.abs(weights[:, k]) / max_w) > weight_threshold)[0]
                module_mets = wide_data.columns[active].tolist()
                if module_mets: modules.append((f"MEFISTO_Factor_{k+1}", module_mets))
        
        Y_hat = np.expm1(factors @ weights.T)
        recon_wide = pd.DataFrame(Y_hat, index=wide_data.index, columns=wide_data.columns)
        preds = recon_wide.melt(var_name="metabolite_id", value_name="fitted_value", ignore_index=False).reset_index()
        preds = preds.merge(sample_meta, on="sample_id", how="left")[["subject_id", "time", "metabolite_id", "fitted_value"]]
        return modules, preds
    except (Exception, SystemExit) as e:
        print(f"  MEFISTO failed: {e}")
    return [], pd.DataFrame()


def analyze_with_dpgp(data: pd.DataFrame, n_grid_points: int = 20, verbose: bool = True) -> Tuple[List[Tuple[str, List[str]]], pd.DataFrame]:
    if verbose: print("Running DPGP analysis...")
    try:
        from scipy.interpolate import interp1d
        from sklearn.mixture import BayesianGaussianMixture
        data_copy = data.copy()
        data_copy["value"] = np.log1p(data_copy["value"])
        common_time = np.linspace(data_copy["time"].min(), data_copy["time"].max(), n_grid_points)
        interp_mat, valid_mets = [], []
        for met, sub in data_copy.groupby("metabolite_id"):
            mean_by_time = sub.groupby("time")["value"].mean()
            if len(mean_by_time) < 2: continue
            y = interp1d(mean_by_time.index, mean_by_time.values, fill_value="extrapolate")(common_time)
            if np.std(y) > 0: y = (y - np.mean(y)) / np.std(y)
            interp_mat.append(y); valid_mets.append(met)
        X = np.array(interp_mat)
        if len(X) == 0: return [], pd.DataFrame()
        labels = BayesianGaussianMixture(n_components=20, random_state=42).fit_predict(X)
        modules = []
        for l in np.unique(labels):
            mets = [valid_mets[i] for i in np.where(labels == l)[0]]
            if len(mets) >= 5: modules.append((f"DPGP_Cluster_{l+1}", mets))
        return modules, calculate_centroid_predictions(data, modules)
    except (Exception, SystemExit) as e:
        print(f"  DPGP failed: {e}")
    return [], pd.DataFrame()


def fit_metabolite_lmms(data: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if verbose: print("Fitting LMMs...")
    try:
        import warnings

        from statsmodels.formula.api import mixedlm
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        data_mod = data.copy()
        data_mod["log_value"] = np.log1p(data_mod["value"])
        results, all_preds = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            with suppress_output(not verbose):
                for mid, mdata in data_mod.groupby("metabolite_id"):
                    try:
                        fit = mixedlm("log_value ~ time", mdata, groups=mdata["subject_id"]).fit(method="powell", disp=False)
                        results.append({"metabolite_id": mid, "pvalue": fit.pvalues.get("time", 1.0), "tstat": fit.tvalues.get("time", 0.0)})
                        p = mdata[["subject_id", "time", "metabolite_id"]].copy()
                        p["fitted_value"] = np.expm1(fit.predict(mdata))
                        all_preds.append(p)
                    except Exception:
                        results.append({"metabolite_id": mid, "pvalue": 1.0, "tstat": 0.0})
        return pd.DataFrame(results), pd.concat(all_preds) if all_preds else pd.DataFrame()
    except (Exception, SystemExit) as e:
        print(f"  LMM fitting failed: {e}")
    return pd.DataFrame(), pd.DataFrame()

def analyze_with_lmm_ora(lmm_results: pd.DataFrame, annotated_pathways: Dict[str, List[str]], pvalue_threshold: float = 0.05) -> List[str]:
    from scipy.stats import hypergeom
    if lmm_results.empty: return []
    sig = lmm_results[lmm_results["pvalue"] < pvalue_threshold]["metabolite_id"].unique()
    all_mets = lmm_results["metabolite_id"].unique()
    enriched = []
    for pid, p_mets in annotated_pathways.items():
        overlap = len(set(p_mets).intersection(sig))
        if hypergeom.sf(overlap - 1, len(all_mets), len(p_mets), len(sig)) < 0.05: enriched.append(pid)
    return enriched

def analyze_with_mogp_ora(
    modules: List[Tuple[str, List[str]]],
    annotated_pathways: Dict[str, List[str]],
    all_metabolite_ids: List[str],
    pvalue_threshold: float = 0.05,
) -> List[str]:
    """Hypergeometric ORA on each MOGP module against each annotated pathway.

    For each module, runs a one-sided hypergeometric test per pathway. The
    per-pathway p-value is the minimum across all modules, Bonferroni-corrected
    for the number of modules tested. A pathway is detected if its corrected
    p-value < pvalue_threshold.
    """
    from scipy.stats import hypergeom

    if not modules:
        return []
    N = len(all_metabolite_ids)
    n_modules = len(modules)
    bonferroni_threshold = pvalue_threshold / n_modules
    pathway_min_p: Dict[str, float] = {pid: 1.0 for pid in annotated_pathways}
    for _, mod_mets in modules:
        mod_set = set(mod_mets)
        n = len(mod_set)
        for pid, p_mets in annotated_pathways.items():
            K = len(p_mets)
            k = len(mod_set.intersection(p_mets))
            if k == 0:
                continue
            p = hypergeom.sf(k - 1, N, K, n)
            if p < pathway_min_p[pid]:
                pathway_min_p[pid] = p
    return [pid for pid, p in pathway_min_p.items() if p < bonferroni_threshold]


def analyze_with_mogp_gsea(
    W_df: pd.DataFrame,
    annotated_pathways: Dict[str, List[str]],
    pvalue_threshold: float = 0.05,
) -> List[str]:
    """Preranked GSEA on MOGP absolute loading weights per factor.

    For each factor k, metabolites are ranked by |W_{i,k}| and gseapy.prerank
    tests enrichment of each annotated pathway.  The per-pathway p-value is the
    minimum across all factors, Bonferroni-corrected for the number of factors
    tested.  A pathway is detected if its corrected p-value < pvalue_threshold.
    """
    try:
        import gseapy as gp
    except ImportError:
        print("  MOGP-GSEA skipped: gseapy not installed")
        return []

    if W_df.empty:
        return []

    n_factors = len(W_df.columns)
    bonferroni_threshold = pvalue_threshold / n_factors
    pathway_min_p: Dict[str, float] = {pid: 1.0 for pid in annotated_pathways}
    for col in W_df.columns:
        rnk = W_df[col].abs().sort_values(ascending=False)
        rnk_df = rnk.reset_index()
        rnk_df.columns = ["metabolite_id", "abs_loading"]
        try:
            res = gp.prerank(
                rnk=rnk_df,
                gene_sets=annotated_pathways,
                threads=1,
                outdir=None,
                min_size=1,
                max_size=10000,
                permutation_num=1000,
                seed=42,
                verbose=False,
            )
            for term in res.res2d["Term"].tolist():
                row = res.res2d[res.res2d["Term"] == term]
                nom_p = float(row["NOM p-val"].values[0])
                if term in pathway_min_p and nom_p < pathway_min_p[term]:
                    pathway_min_p[term] = nom_p
        except Exception:
            continue

    return [pid for pid, p in pathway_min_p.items() if p < bonferroni_threshold]


def analyze_with_lmm_gsea(
    lmm_results: pd.DataFrame, annotated_pathways: Dict[str, List[str]]
) -> List[str]:
    # Preranked GSEA using |t-stat| from LMM time coefficient.
    # Absolute values used so that both LMM+GSEA and MOGP+GSEA test the same
    # question: "is this pathway enriched for metabolites with strong temporal
    # dynamics?" — making the two directly comparable. Signed t-stats would
    # test coordinated unidirectional regulation, but pathway_random_effect_sd
    # creates bidirectional effects that violate that assumption.
    #
    # Threshold: NOM p-val < 0.05 per pathway (not FDR q-val), consistent with
    # LMM+ORA and MOGP+GSEA (all per-pathway alpha = 0.05). FDR is not applied
    # because with only 5 pathways gseapy's permutation-based FDR is poorly
    # calibrated.
    try:
        import gseapy as gp
    except ImportError:
        print("  LMM-GSEA skipped: gseapy not installed (pip install gseapy)")
        return []
    try:
        rnk = lmm_results[["metabolite_id", "tstat"]].copy()
        rnk["tstat"] = rnk["tstat"].abs()
        rnk = rnk.sort_values("tstat", ascending=False)
        res = gp.prerank(
            rnk=rnk,
            gene_sets=annotated_pathways,
            threads=1,
            outdir=None,
            seed=9102,
            verbose=False,
            min_size=5,
        )
        nom_col = "NOM p-val" if "NOM p-val" in res.res2d.columns else "pval"
        return res.res2d[res.res2d[nom_col] < 0.05]["Term"].tolist()
    except (Exception, SystemExit) as e:
        print(f"  LMM-GSEA failed: {e}")
    return []

def analyze_with_meba(data: pd.DataFrame, annotated_pathways: Dict[str, List[str]], verbose: bool = True) -> Tuple[List[str], pd.DataFrame]:
    if verbose: print("Running MEBA analysis...")
    if not ensure_r_dependencies(["Rcpp", "Rserve", "lme4"], github_packages=["xia-lab/MetaboAnalystR"]): return [], pd.DataFrame()
    try:
        import tempfile

        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        df_log = data.copy(); df_log["value"] = np.log1p(df_log["value"])
        # Group variable (real covariate or artificial even split).
        if "group" in data.columns and data["group"].nunique() >= 2:
            df_log["Group"] = data["group"].astype(str)
        else:
            subjects = sorted(df_log["subject_id"].unique())
            mid = max(1, len(subjects) // 2)
            group_map = {s: "G1" if i < mid else "G2" for i, s in enumerate(subjects)}
            df_log["Group"] = df_log["subject_id"].map(group_map)
        # Rank-based time bins (handles irregular sampling).
        # Rank only UNIQUE (subject_id, time) pairs — df_log is in long format
        # with one row per (subject, metabolite, time), so a naive groupby-rank
        # would assign distinct ranks to each metabolite row at the same timepoint.
        unique_tp = df_log[["subject_id", "time"]].drop_duplicates().copy()
        unique_tp["time_rank"] = unique_tp.groupby("subject_id")["time"].rank(method="first").astype(int)
        df_log = df_log.merge(unique_tp, on=["subject_id", "time"], how="left")
        n_tp = int(df_log["time_rank"].max())
        pad = len(str(n_tp))
        df_log["time_label"] = "T" + df_log["time_rank"].apply(lambda x: str(int(x)).zfill(pad))
        wide = df_log.pivot_table(
            index=["subject_id", "Group", "time_label"], columns="metabolite_id",
            values="value", aggfunc="first",
        ).reset_index()
        wide["Sample"] = wide["subject_id"].astype(str) + "_" + wide["time_label"]
        metab_cols = [c for c in wide.columns if c.startswith("M") and c[1:].isdigit()]
        # Main data CSV: Sample (col 1), Group/class (col 2), metabolites only.
        # Time and subject_id are excluded — MetaboAnalystR would treat them as
        # metabolite features.  Time is provided via a separate metadata file so
        # performMB can run the full 2-factor (group × time) MB analysis.
        data_wide = wide[["Sample", "Group"] + metab_cols]
        # Metadata CSV (for ReadMetaData): Sample, Time (col 1 → time.fac),
        # Group (col 2 → exp.fac).  Column order matters when meta.vec.mb is empty.
        meta_df = wide[["Sample", "time_label", "Group"]].rename(columns={"time_label": "Time"})
        # Create an isolated run directory in Python so each parallel worker
        # and each successive call gets its own scratch space.  MetaboAnalystR
        # writes/reads intermediate .qs files by relative path, so the working
        # directory must be stable and unique for the entire pipeline call.
        run_dir = tempfile.mkdtemp(prefix="meba_run_")
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, dir=run_dir) as tmp:
                data_wide.to_csv(tmp.name, index=False); tmp_path = tmp.name
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, dir=run_dir) as tmeta:
                meta_df.to_csv(tmeta.name, index=False); meta_path = tmeta.name
            ro.r("""
                library(MetaboAnalystR)
                run_meba <- function(p, meta_p, run_dir) {
                    old_wd <- getwd()
                    on.exit(setwd(old_wd))
                    setwd(run_dir)
                    if (exists("mSet")) rm(mSet, envir = .GlobalEnv)
                    mSet <- InitDataObjects(data.type="conc", anal.type="ts", paired=FALSE, default.dpi=72)
                    mSet <- Read.TextData(mSet, p, "rowu", "disc")
                    mSet <- SanityCheckData(mSet)
                    # Load Time + Group metadata so performMB can use the 2-factor
                    # (group x time) MB analysis path.
                    mSet <- ReadMetaData(mSet, meta_p)
                    FilterVariable(mSet, qc.filter="F", var.filter="none")
                    # "min" is not a recognised method in this MetaboAnalystR
                    # version; use "colmin" (1/2 of per-feature column min).
                    mSet <- ImputeMissingVar(mSet, method="colmin")
                    mSet <- PreparePrenormData(mSet)
                    mSet <- Normalization(mSet, "NULL", "NULL", "NULL", ratio=FALSE, ratioNum=20)
                    mSet <- SetDesignType(mSet, "time")
                    # performMB requires meta.info columns to be *factors*; ReadMetaData
                    # reads the CSV → character type so levels() returns NULL → time.len=0
                    # → zero-extent size matrix → "incorrect sample sizes" error.
                    # Convert here.  Fall back to parsing sample names if ReadMetaData
                    # did not populate meta.info (e.g. sample-name mismatch).
                    if (!is.null(mSet$dataSet$meta.info) && nrow(mSet$dataSet$meta.info) > 0) {
                        mi <- mSet$dataSet$meta.info
                        if ("Time" %in% colnames(mi))
                            mi[["Time"]] <- factor(mi[["Time"]], levels = sort(unique(as.character(mi[["Time"]]))))
                        if ("Group" %in% colnames(mi))
                            mi[["Group"]] <- factor(mi[["Group"]])
                        mSet$dataSet$meta.info <- mi
                    } else {
                        smpl <- rownames(mSet$dataSet$norm)
                        tlab <- sub(".*_(T[0-9]+)$", "\\1", smpl)
                        glab <- as.character(mSet$dataSet$cls)
                        mSet$dataSet$meta.info <- data.frame(
                            Time  = factor(tlab,  levels = sort(unique(tlab))),
                            Group = factor(glab),
                            row.names = smpl
                        )
                    }
                    # performMB looks up meta.vec.mb through the MetaboAnalystR
                    # namespace scope chain; run_meba local env is invisible to it.
                    # Must use <<- to assign in the global env.
                    meta.vec.mb <<- c("Time", "Group")
                    mSet <- tryCatch(performMB(mSet), error=function(e) { message("performMB failed: ", e$message); NULL })
                    if (is.null(mSet) || is.null(mSet$analSet$MB$stats)) return(character(0))
                    rownames(mSet$analSet$MB$stats)[order(mSet$analSet$MB$stats[,1], decreasing=TRUE)]
                }
            """)
            ranked = list(ro.r(f'run_meba("{tmp_path}", "{meta_path}", "{run_dir}")'))
            top_mets = ranked[:max(1, int(len(ranked)*0.1))]
            enriched = [pid for pid, p_mets in annotated_pathways.items() if len(set(top_mets).intersection(p_mets)) >= 2]
            return enriched, calculate_centroid_predictions(data, [("MEBA_Top", top_mets)])
        finally:
            import shutil
            shutil.rmtree(run_dir, ignore_errors=True)
    except (Exception, SystemExit) as e:
        print(f"  MEBA failed: {e}")
    return [], pd.DataFrame()

def analyze_with_timeomics(data: pd.DataFrame, verbose: bool = True) -> Tuple[List[Tuple[str, List[str]]], pd.DataFrame]:
    if verbose: print("Running timeOmics analysis...")
    if not ensure_r_dependencies(["dplyr"], bioc_packages=["timeOmics", "mixOmics"], github_packages=["cran/lmms"]): return [], pd.DataFrame()
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter

        data_log = data.copy()
        data_log["value"] = np.log1p(data_log["value"])
        data_log = data_log.sort_values(["metabolite_id", "subject_id", "time"])

        keepX_val = max(1, int(data["metabolite_id"].nunique() / 5))

        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["df_long"] = ro.conversion.py2rpy(data_log)
        ro.globalenv["keepX_val"] = keepX_val

        ro.r("""
            library(timeOmics); library(mixOmics); library(lmms)
            run_timeomics <- function(df, keepX_val) {
                unique_mets <- unique(df$metabolite_id)
                n_mets <- length(unique_mets)
                n_obs <- nrow(df) / n_mets
                if (n_obs %% 1 != 0) return(NULL)

                mat <- matrix(df$value, nrow=n_obs, ncol=n_mets, byrow=FALSE)
                colnames(mat) <- unique_mets
                time_vec <- as.numeric(df$time[1:n_obs])
                sampleID_vec <- as.character(df$subject_id[1:n_obs])

                grid_points <- 10
                pred_time <- seq(min(time_vec), max(time_vec), length.out=grid_points)
                models <- tryCatch({
                    suppressWarnings(utils::capture.output({
                        res <- lmms::lmmSpline(data=mat, time=time_vec, sampleID=sampleID_vec,
                                               timePredict=pred_time, keepModels=TRUE, numCores=1)
                    }))
                    res
                }, error = function(e) NULL)
                if (is.null(models)) return(NULL)

                # predSpline is a data frame with one list-column; each element is a
                # numeric vector of grid_points predicted values for one metabolite.
                # Bind into a numeric n_mets x grid_points matrix, then transpose
                # to get grid_points x n_mets (samples x features) as spls expects.
                X_mat <- do.call(rbind, lapply(models@predSpline[[1]], as.numeric))
                X_mat[is.na(X_mat)] <- 0
                rownames(X_mat) <- head(colnames(mat), nrow(X_mat))
                X <- t(X_mat)

                Y <- matrix(1:nrow(X), ncol=1)
                res_spls <- tryCatch({
                    mixOmics::spls(X=X, Y=Y, ncomp=1, keepX=keepX_val, mode='regression', scale=TRUE)
                }, error = function(e) NULL)
                if (is.null(res_spls)) return(NULL)

                clusters <- tryCatch({ timeOmics::getCluster(res_spls) }, error = function(e) NULL)
                if (is.null(clusters) || nrow(clusters) == 0) {
                    loadings <- res_spls$loadings$X[,1]
                    active <- names(loadings)[abs(loadings) > 1e-10]
                    if (length(active) == 0) return(NULL)
                    cluster_ids <- ifelse(loadings[active] >= 0, "1", "-1")
                    all_ids <- setNames(rep("grey", length(loadings)), names(loadings))
                    all_ids[active] <- cluster_ids
                    return(all_ids)
                }
                setNames(as.character(clusters$cluster), clusters$molecule)
            }
        """)
        r_clusters = ro.r("run_timeomics(df_long, keepX_val)")
        if r_clusters is None or not hasattr(r_clusters, 'names'): return [], pd.DataFrame()
        cluster_map = dict(zip(r_clusters.names, list(r_clusters)))
        modules = {}
        for m, c in cluster_map.items(): modules.setdefault(f"timeOmics_Cluster_{c}", []).append(str(m))
        mod_list = list(modules.items())
        if verbose: print(f"  timeOmics found {len(mod_list)} clusters.")
        return mod_list, calculate_centroid_predictions(data, mod_list)
    except (Exception, SystemExit) as e:
        if verbose: print(f"  timeOmics failed: {repr(e)}")
    return [], pd.DataFrame()

def analyze_with_pal(
    data: pd.DataFrame,
    annotated_pathways: Dict[str, List[str]],
    p_val_threshold: float = 0.05,
    verbose: bool = True,
) -> Tuple[List[str], pd.DataFrame]:
    """Analyzes data with PAL using custom pathway injection."""
    if verbose: print("Running PAL analysis...")
    if not ensure_r_dependencies(["PAL", "igraph", "lme4", "PASI"]): return [], pd.DataFrame()

    pathway_dir = tempfile.mkdtemp()
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        
        data_log = data.copy()
        data_log["value"] = np.log1p(data_log["value"])
        data_log["sample_id"] = data_log.apply(lambda r: get_standardized_sample_id(r["subject_id"], r["time"]), axis=1)
        
        all_mets = data_log["metabolite_id"].unique()
        metab_to_num = {m: str(i+1000) for i, m in enumerate(all_mets)}
        data_log["metab_num"] = data_log["metabolite_id"].map(metab_to_num)
        
        wide = data_log.pivot(index="metab_num", columns="sample_id", values="value").fillna(0)
        
        active_p = annotated_pathways.copy()
        if len(active_p) == 1:
            active_p["Dummy"] = [m for m in all_mets if m not in active_p[list(active_p.keys())[0]]][:5]

        for pid, mets in active_p.items():
            num_mets = [metab_to_num[m] for m in mets if m in metab_to_num]
            if len(num_mets) < 2: continue
            with open(os.path.join(pathway_dir, f"{pid}.txt"), "w") as f:
                f.write(f"{pid}\n")
                for m in num_mets: f.write(f"{m}\n")
                for i in range(len(num_mets)-1): f.write(f"{num_mets[i]} {num_mets[i+1]} +\n")

        info = data_log[["sample_id", "subject_id", "time"]].drop_duplicates().set_index("sample_id").reindex(wide.columns)
        if "group" in data_log.columns:
            group_map = data_log[["sample_id", "group"]].drop_duplicates().set_index("sample_id")["group"]
            info["group"] = info.index.map(group_map).fillna(0).astype(int)
        else:
            info["group"] = 0

        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["df_pal_data"] = ro.conversion.py2rpy(wide)
            ro.globalenv["df_pal_info"] = ro.conversion.py2rpy(info)
            ro.globalenv["path_dir"] = pathway_dir
            ro.globalenv["debug_mode"] = bool(verbose)

        ro.r("""
            library(PAL); library(PASI)
            run_pal <- function(d, i, p_dir, debug) {
                old_wd <- getwd()
                # If not in debug mode, change WD to temp dir to catch PAL_date.txt
                if (!debug) setwd(p_dir)
                on.exit(setwd(old_wd))

                assign("neutralise", NULL, envir = .GlobalEnv)  # PAL body references this; assign globally so PAL's lexical scope can find it
                res <- tryCatch({
                    suppressWarnings(PAL(data=d, info=i, grouplabels="group", mainfeature="time",
                        userandom="subject_id", useKEGG=FALSE,
                        pathwayadress=p_dir, nodemin=2))
                }, error = function(e) NULL)
                
                if (is.null(res) || length(res) < 2) return(NULL)
                pvals <- res[[2]][, grep("^Pval_", colnames(res[[2]]))[1]]
                names(pvals) <- rownames(res[[2]])
                pvals
            }
        """)
        p_vals = ro.r("run_pal(df_pal_data, df_pal_info, path_dir, debug_mode)")
        
        significant, p_map = [], {}
        if p_vals is not None and p_vals != ro.rinterface.NULL:
            p_map_raw = dict(zip(p_vals.names, list(p_vals)))
            for k, v in p_map_raw.items():
                name = k.split(":")[0]
                if name in annotated_pathways: p_map[name] = v
            significant = [pid for pid, p in p_map.items() if p < p_val_threshold]

        best_pid = (
            significant[0]
            if significant
            else (min(p_map, key=p_map.get) if p_map else None)
        )
        sig_mods = (
            [(f"PAL_{best_pid}", annotated_pathways[best_pid])] if best_pid else []
        )
        return significant, calculate_centroid_predictions(data, sig_mods)
    except (Exception, SystemExit) as e:
        print(f"  PAL failed: {e}")
    finally:
        if os.path.exists(pathway_dir):
            shutil.rmtree(pathway_dir)
    return [], pd.DataFrame()
