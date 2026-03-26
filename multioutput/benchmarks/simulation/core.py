"""
Core simulation engine for generating realistic longitudinal metabolomics data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable
from .effects import negative_binomial_likelihood, create_spike_effect

def simulate_longitudinal_data(
    n_subjects: int = 10,
    n_metabolites: int = 100,
    n_pathways: int = 5,
    metabolites_per_pathway: int = 20,
    time_points: np.ndarray = np.linspace(0, 20, 15),
    add_group_covariate: bool = False,
    irregular_sampling_sd: float = 1.5,
    subject_random_effect_sd: float = 0.2,
    pathway_random_effect_sd: float = 0.3,
    metabolite_baseline_sd: float = 0.5,
    metabolite_effect_sd: float = 0.2,
    active_pathway_idx: int = 0,
    effect_func: Callable[[np.ndarray], np.ndarray] = None,
    nuisance_fraction: float = 0.2,
    nuisance_pathway_only: bool = False,
    nuisance_effect_func: Callable[[np.ndarray], np.ndarray] = None,
    likelihood_func: Callable[[np.ndarray, float], np.ndarray] = negative_binomial_likelihood,
    dispersion: float = 10.0,
    annotation_fraction: float = 0.7,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, List[str]]]:
    """Generates a realistic, complex longitudinal metabolomics dataset."""
    
    if effect_func is None:
        effect_func = create_spike_effect(10, 12, 2.0)

    # --- Setup Metabolites and Pathways ---
    metabolite_ids = [f"M{i+1:03d}" for i in range(n_metabolites)]
    pathway_ids = [f"Pathway_{i+1}" for i in range(n_pathways)]

    true_pathway_mapping = {}
    for i, pid in enumerate(pathway_ids):
        start_idx = i * metabolites_per_pathway
        end_idx = start_idx + metabolites_per_pathway
        true_pathway_mapping[pid] = metabolite_ids[start_idx:end_idx]

    annotated_pathway_mapping = {}
    for pid, metabolites in true_pathway_mapping.items():
        n_annotated = int(len(metabolites) * annotation_fraction)
        annotated_metabolites = np.random.choice(metabolites, n_annotated, replace=False).tolist()
        annotated_pathway_mapping[pid] = annotated_metabolites

    # --- Setup Covariates and Random Effects ---
    subject_intercepts = np.random.normal(0, subject_random_effect_sd, n_subjects)
    subject_group = np.random.randint(0, 2, n_subjects)
    metabolite_baselines = np.random.normal(0, metabolite_baseline_sd, n_metabolites)
    pathway_intercepts = np.random.normal(0, pathway_random_effect_sd, (n_subjects, n_pathways))

    active_pathway_metabolites = set(true_pathway_mapping[pathway_ids[active_pathway_idx]])
    is_active_metab = np.array([m in active_pathway_metabolites for m in metabolite_ids])

    metabolite_scalers = np.ones(n_metabolites)
    metabolite_scalers[is_active_metab] = np.random.normal(0, metabolite_effect_sd, size=np.sum(is_active_metab))

    # --- Generate Time Points ---
    n_tp = len(time_points)
    base_times = np.tile(time_points, (n_subjects, 1))
    time_noise = np.random.normal(0, irregular_sampling_sd, (n_subjects, n_tp))
    subject_times = np.sort(np.maximum(0, base_times + time_noise), axis=1)

    # --- Generate Latent Signal ---
    latent_signal = (
        subject_intercepts[:, np.newaxis, np.newaxis]
        + metabolite_baselines[np.newaxis, np.newaxis, :]
        + np.zeros((n_subjects, n_tp, n_metabolites))
    )

    for p_idx, pid in enumerate(pathway_ids):
        p_mets = true_pathway_mapping[pid]
        met_indices = [metabolite_ids.index(m) for m in p_mets]
        latent_signal[:, :, met_indices] += pathway_intercepts[:, np.newaxis, p_idx, np.newaxis]

    temporal_profile = effect_func(subject_times)
    if add_group_covariate:
        group_mod = np.where(subject_group == 1, 0.5, 1.0)[:, np.newaxis]
        temporal_profile = temporal_profile * group_mod

    active_indices = np.where(is_active_metab)[0]
    if len(active_indices) > 0:
        latent_signal[:, :, active_indices] += (
            temporal_profile[:, :, np.newaxis] * metabolite_scalers[np.newaxis, np.newaxis, active_indices]
        )

    if nuisance_fraction > 0 and nuisance_effect_func is not None:
        if nuisance_pathway_only:
            # Nuisance (circadian) oscillations concentrated in annotated pathway
            # metabolites, not unannotated background metabolites.
            candidate_indices = sorted({
                metabolite_ids.index(m)
                for mets in true_pathway_mapping.values()
                for m in mets
            })
        else:
            candidate_indices = list(range(n_metabolites))
        n_nuisance = min(int(len(candidate_indices) * nuisance_fraction), len(candidate_indices))
        if n_nuisance > 0:
            nuisance_indices = np.random.choice(candidate_indices, n_nuisance, replace=False)
            nuisance_profile = nuisance_effect_func(subject_times)
            latent_signal[:, :, nuisance_indices] += nuisance_profile[:, :, np.newaxis]

    # --- Generate Observations ---
    mu = np.exp(latent_signal)
    observed_values = likelihood_func(mu, dispersion)

    # --- Flatten and Construct DataFrame ---
    subj_ids_str = [f"S_{i+1}" for i in range(n_subjects)]
    col_subj = np.repeat(subj_ids_str, n_tp * n_metabolites)
    col_time = np.repeat(subject_times.flatten(), n_metabolites)
    col_metab = np.tile(metabolite_ids, n_subjects * n_tp)
    col_val = observed_values.flatten()
    col_mu = mu.flatten()
    col_is_active = np.tile(is_active_metab.astype(int), n_subjects * n_tp)

    data_dict = {
        "subject_id": col_subj, "metabolite_id": col_metab, "time": col_time,
        "value": col_val, "true_mu": col_mu, "is_active": col_is_active,
    }
    if add_group_covariate:
        data_dict["group"] = np.repeat(subject_group, n_tp * n_metabolites)

    return pd.DataFrame(data_dict), true_pathway_mapping, annotated_pathway_mapping
