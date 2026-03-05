"""
Shared utilities and evaluation metrics for the longitudinal simulation framework.
"""

import os
import sys
import logging
import contextlib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any

# =============================================================================
# 1. Environment and Logging Setup
# =============================================================================

def setup_environment():
    """Sets environment variables to limit thread usage and silence logs."""
    # Point rpy2 to the R installed in this conda environment (not the base env).
    # rpy2's auto-discovery can pick up a different R if R_HOME is not set.
    r_home = os.path.join(sys.prefix, "lib", "R")
    if os.path.isdir(r_home):
        os.environ.setdefault("R_HOME", r_home)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # Silence TensorFlow and absl
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except (ImportError, Exception):
        pass

    # Silence rpy2 logging and install resilient console callbacks.
    # In worker processes (ProcessPoolExecutor), Python finalizers can close
    # sys.stderr before rpy2 cleanup runs, causing "I/O operation on closed file".
    # Replacing the callbacks with error-tolerant versions prevents this.
    try:
        import rpy2.rinterface_lib.callbacks as _rcb

        def _safe_r_write(x):
            try:
                sys.stderr.write(x)
            except (ValueError, OSError, AttributeError, TypeError):
                pass
        _rcb.consolewrite_warnerror = _safe_r_write
        _rcb.consolewrite_print = _safe_r_write
        logger = logging.getLogger("rpy2.rinterface_lib.callbacks")
        logger.setLevel(logging.ERROR)
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
    except ImportError:
        pass

@contextlib.contextmanager
def suppress_output(suppress=True):
    """
    Context manager to redirect stdout and stderr to devnull at the Python level.
    """
    if not suppress:
        yield
        return

    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# =============================================================================
# 2. R Dependency Management
# =============================================================================

def ensure_r_dependencies(
    packages: List[str],
    bioc_packages: List[str] = None,
    github_packages: List[str] = None,
):
    """Ensures that required R packages are installed via rpy2."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        utils = importr("utils")
        utils.chooseCRANmirror(ind=1)

        for pkg in packages:
            try:
                if pkg == "igraph":
                    importr(pkg, on_conflict="warn", robject_translations={".env": "__env_igraph"})
                else:
                    importr(pkg, on_conflict="warn")
            except (ImportError, Exception):
                utils.install_packages(pkg)

        if bioc_packages:
            try:
                importr("BiocManager")
            except ImportError:
                utils.install_packages("BiocManager")
            bioc_mgr = importr("BiocManager")
            for pkg in bioc_packages:
                try:
                    importr(pkg)
                except ImportError:
                    bioc_mgr.install(pkg, update=False, ask=False)

        if github_packages:
            try:
                importr("remotes")
            except ImportError:
                utils.install_packages("remotes")
            remotes = importr("remotes")
            for pkg in github_packages:
                pkg_name = pkg.split("/")[-1]
                try:
                    importr(pkg_name)
                except ImportError:
                    remotes.install_github(pkg, upgrade="never")
        return True
    except Exception as e:
        print(f"  Error setting up R dependencies: {e}")
        return False

# =============================================================================
# 3. Evaluation Metrics
# =============================================================================

def get_standardized_sample_id(subject_id: str, time: float) -> str:
    """Generates a consistent sample ID from subject and time."""
    return f"{subject_id}_{float(time):.4f}"

def calculate_reconstruction_fidelity(
    fitted_df: pd.DataFrame, true_metabolites: set
) -> float:
    """Calculates the MSE between true latent signal and predicted signal."""
    active_df = fitted_df[fitted_df['metabolite_id'].isin(true_metabolites)]
    if active_df.empty or 'true_mu' not in active_df.columns or 'fitted_value' not in active_df.columns:
        return 0.0
    log_true = np.log1p(active_df['true_mu'])
    log_fitted = np.log1p(active_df['fitted_value'])
    return np.mean((log_true - log_fitted) ** 2)

def evaluate_module(
    module_metabolites: List[str],
    true_pathway_metabolites: set,
    unannotated_in_pathway: set,
    verbose: bool = False,
) -> Dict[str, float]:
    """Calculates Jaccard, Precision, and Recall for unannotated metabolites."""
    module_set = set(module_metabolites)
    true_set = true_pathway_metabolites
    intersection = len(module_set.intersection(true_set))
    union = len(module_set.union(true_set))
    jaccard = intersection / union if union > 0 else 0
    precision = intersection / len(module_set) if len(module_set) > 0 else 0
    found_unannotated_count = len(module_set.intersection(unannotated_in_pathway))
    total_unannotated = len(unannotated_in_pathway)
    unannotated_recall = found_unannotated_count / total_unannotated if total_unannotated > 0 else 0.0

    if verbose:
        print(f"    - Jaccard Index: {jaccard:.2f}, Precision: {precision:.2f}, Unannotated Recall: {unannotated_recall:.2f}")

    return {
        "jaccard": jaccard,
        "precision": precision,
        "unannotated_found": found_unannotated_count,
        "unannotated_recall": unannotated_recall,
    }

def evaluate_clustering_performance(
    true_pathways, predicted_modules, all_metabolite_ids, method_name, verbose=True
):
    """Calculates Adjusted Rand Index (ARI) for clustering methods."""
    from sklearn.metrics import adjusted_rand_score
    t_map = {m: i for i, (pid, mets) in enumerate(true_pathways.items()) for m in mets}
    p_map = {m: i for i, (mn, mets) in enumerate(predicted_modules) for m in mets}
    score = adjusted_rand_score(
        [t_map.get(m, -1) for m in all_metabolite_ids],
        [p_map.get(m, -1) for m in all_metabolite_ids],
    )
    if verbose:
        print(f"  {method_name} ARI: {score:.3f}")
    return score
