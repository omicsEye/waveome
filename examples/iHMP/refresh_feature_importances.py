"""Recompute every model's feature_importance_detail under the corrected BIC.

`calc_metric` now differences the ELBO rather than log_posterior_density
(commit 416514a, FINDINGS 21-22), so every log_bf changes. The permutation
run recomputes its own observed statistic through the same code path and is
therefore fine, but `get_significance_table()` reads log_bf and
deviance_explained out of `model.feature_importance_detail`, which was
computed at fit time under the old BIC. Left alone it would pair OLD log_bf
with NEW q-values.

It also matters for components the permutation never touches: only 4 of the
13 (hbi and time_from_max x lin/SE) are permutation-tested. The other nine --
participant_id, site_name, race, sex, general_wellbeing, study_days x2,
age x2 -- get their log_bf *only* from this detail, and they appear in the
manuscript showcase panels.

Refreshing rather than re-fitting, for two reasons. It is ~4.7x faster (21
min at 15-way parallelism vs 97.9 min for a full penalized_optimization,
whose cost is dominated by the num_restart=3 initial fits that the BIC fix
does not affect). And it leaves the fitted models byte-identical, so every
downstream difference is attributable to the BIC correction alone -- a
re-fit would re-run stochastic restarts, and with a 98.2% convergence rate
and documented non-reproducible numerical faults, some models would land
differently and confound the comparison.

    python refresh_feature_importances.py [--dry-run N]
"""
import argparse
import os
import pickle
import shutil
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import ray  # noqa: E402

from waveome.model_search import _component_covariate_names  # noqa: E402
from waveome.utilities import convert_data_to_tensors  # noqa: E402

PKL = "output/ihmp_penalized_fit.pkl"
BACKUP = "output/archive_oldbic_2026-09-01/fit_penalized_models_OLDBIC.pkl"

# Independently measured before writing this script, so the result can be
# checked rather than assumed (same red/green habit that caught earlier bugs).
EXPECT = {
    ("HILp_QI578", "squared_exponential", "time_from_max"): -3.70,
    # a component whose variance sits on VARIANCE_FLOOR contributes nothing
    # to the ELBO, so its log_bf must be exactly the parameter penalty
    # -0.5 * 2 * ln(238) = -5.472
    ("HILp_QI7860", "squared_exponential", "hbi"): -5.47,
}


@ray.remote(max_calls=1, max_retries=5)
def refresh(model, X, y):
    try:
        model.get_feature_importances(
            data=convert_data_to_tensors(X, y.reshape(-1, 1)))
        return (model.feature_importances,
                model.feature_importance_detail), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", type=int, default=0,
                    help="refresh only the first N models and do not save")
    args = ap.parse_args()

    with open(PKL, "rb") as f:
        gps = pickle.load(f)
    names = list(gps.models)
    if args.dry_run:
        names = names[:args.dry_run]
        for k in EXPECT:
            if k[0] not in names:
                names.append(k[0])
    print(f"{len(names)} models to refresh")

    ray.init(include_dashboard=False, configure_logging=False)
    X_ref = ray.put(gps.X.to_numpy())
    jobs = {n: refresh.remote(ray.put(gps.models[n]), X_ref,
                              ray.put(gps.Y[n].to_numpy())) for n in names}

    t0, done, failed = time.time(), 0, {}
    pending = list(jobs.values())
    back = {v: k for k, v in jobs.items()}
    while pending:
        ready, pending = ray.wait(pending, num_returns=min(25, len(pending)))
        for ref in ready:
            name = back[ref]
            res, err = ray.get(ref)
            done += 1
            if err is not None:
                failed[name] = err
                continue
            imps, detail = res
            gps.models[name].feature_importances = imps
            gps.models[name].feature_importance_detail = detail
        el = time.time() - t0
        print(f"  {done}/{len(names)} ({el/60:.1f} min, {len(failed)} failed)",
              flush=True)
    ray.shutdown()

    if failed:
        print(f"\n{len(failed)} FAILED -- not saving:")
        for n, e in list(failed.items())[:5]:
            print(f"  {n}: {e[:110]}")
        raise SystemExit(1)

    # --- verify against independently measured values before saving ---
    print("\nverification:")
    ok = True
    for (met, kt, cov), want in EXPECT.items():
        if met not in gps.models:
            continue
        m = gps.models[met]
        kts, cns = _component_covariate_names(m.kernel_name, gps.feat_names)
        j = [i for i, kc in enumerate(zip(kts, cns)) if kc == (kt, cov)][0]
        got = m.feature_importance_detail[j]["log_bf"]
        good = abs(got - want) < 0.15
        ok &= good
        print(f"  {met} {kt}[{cov}]: got {got:+.2f}, expected {want:+.2f}"
              f"  {'OK' if good else 'MISMATCH'}")
    if not ok:
        raise SystemExit("verification failed -- not saving")

    if args.dry_run:
        print("\ndry run -- not saving")
        return
    if not os.path.exists(BACKUP):
        os.makedirs(os.path.dirname(BACKUP), exist_ok=True)
        shutil.copy2(PKL, BACKUP)
        print(f"\nbacked up old-BIC pickle -> {BACKUP}")
    with open(PKL, "wb") as f:
        pickle.dump(gps, f)
    print(f"wrote refreshed {PKL}")


if __name__ == "__main__":
    main()
