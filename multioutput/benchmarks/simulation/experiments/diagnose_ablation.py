#!/usr/bin/env python3
"""
diagnose_ablation.py — Isolation test for Otsu threshold vs nuisance restriction.

Tests 4 configurations x 5 replicates at medium SNR (spike effect only for speed):
  A: pathway-only nuisance + Otsu       (current production)
  B: uniform nuisance    + Otsu
  C: pathway-only nuisance + fixed 0.2
  D: uniform nuisance    + fixed 0.2    (original baseline)

Run from project root:
    /Users/allen/miniforge3/envs/mogp-waveome-sim/bin/python3.11 \
        code/simulation/experiments/diagnose_ablation.py
"""

import os, sys, time
os.environ["R_HOME"] = "/Users/allen/miniforge3/envs/mogp-waveome-sim/lib/R"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from code.simulation.core import simulate_longitudinal_data
from code.simulation.effects import create_spike_effect, create_periodic_effect
from code.simulation.methods import analyze_with_mogp, analyze_with_mogp_ora

N_REPLICATES = 5
MEDIUM_PARAMS = dict(
    n_subjects=20,
    n_metabolites=200,
    n_pathways=5,
    metabolites_per_pathway=20,
    time_points=np.linspace(0, 20, 5),
    irregular_sampling_sd=1.5,
    subject_random_effect_sd=0.3,
    dispersion=10.0,
    nuisance_fraction=0.2,
    annotation_fraction=0.7,
)

CONFIGS = {
    "A: pathway-nuisance + Otsu":    dict(nuisance_pathway_only=True,  use_otsu=True),
    "B: uniform-nuisance  + Otsu":   dict(nuisance_pathway_only=False, use_otsu=True),
    "C: pathway-nuisance + fixed":   dict(nuisance_pathway_only=True,  use_otsu=False),
    "D: uniform-nuisance  + fixed":  dict(nuisance_pathway_only=False, use_otsu=False),
}

effect_func = create_spike_effect(10, 12, 2.5)
nuisance_func = create_periodic_effect(1.0, 10.0)

print("Running ablation (5 replicates × 4 configs, spike effect, medium SNR)...\n")

results = {}
for cfg_name, cfg in CONFIGS.items():
    sens_list, fpr_list = [], []
    t0 = time.time()
    for seed in range(42, 42 + N_REPLICATES):
        np.random.seed(seed)
        data, true_pathways, annotated_pathways = simulate_longitudinal_data(
            **MEDIUM_PARAMS,
            effect_func=effect_func,
            nuisance_effect_func=nuisance_func,
            nuisance_pathway_only=cfg["nuisance_pathway_only"],
        )
        all_ids = sorted(data["metabolite_id"].unique())
        active_pid = list(true_pathways.keys())[0]

        modules, _, _ = analyze_with_mogp(
            data, use_otsu=cfg["use_otsu"], verbose=False
        )
        detected = analyze_with_mogp_ora(modules, annotated_pathways, all_ids)

        n_null = len(annotated_pathways) - 1
        tp = 1 if active_pid in detected else 0
        fp = sum(1 for p in detected if p != active_pid)
        sens_list.append(float(tp))
        fpr_list.append(float(fp) / n_null if n_null > 0 else 0.0)

    elapsed = time.time() - t0
    results[cfg_name] = dict(
        sens=np.mean(sens_list),
        fpr=np.mean(fpr_list),
        sens_std=np.std(sens_list),
        fpr_std=np.std(fpr_list),
        time_s=elapsed,
    )
    print(f"  {cfg_name}")
    print(f"    Sensitivity: {results[cfg_name]['sens']:.3f} ± {results[cfg_name]['sens_std']:.3f}")
    print(f"    FPR:         {results[cfg_name]['fpr']:.3f} ± {results[cfg_name]['fpr_std']:.3f}")
    print(f"    Time:        {elapsed:.0f}s\n")

print("\n=== SUMMARY ===")
print(f"{'Config':<35} {'Sens':>6} {'FPR':>6}")
print("-" * 50)
for cfg_name, r in results.items():
    print(f"{cfg_name:<35} {r['sens']:>6.3f} {r['fpr']:>6.3f}")
