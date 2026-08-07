"""Refresh feature_importance_detail for every model in the no-prune fit,
using the current (clamp-branch-removed) calc_feature_importance_components.

The stored no_prune pickle's near-floor SE components are systematically
wrong: the old clamp shortcut always charged p=1 regardless of kernel type,
but squared_exponential really costs p=2 (variance + lengthscale) -- so
every near-floor SE component is sitting at the p=1 (lin-appropriate)
reference value instead of its own correct one. This recomputes every
component for real via genuine refits (no more clamp shortcut at all),
parallelized via Ray to avoid a multi-hour sequential run.

Does NOT re-run kernel search or model fitting -- the existing model
structures and fitted parameters are already correct and are left
untouched. Only get_feature_importances(refit=True) is re-run per model.

Run from examples/iHMP/ (same working directory as the notebook):
    python refresh_significance_no_prune.py
"""
import pickle
import time

import ray

from waveome.utilities import convert_data_to_tensors

INPUT_FP = "output/fit_penalized_models_revision_full_scipy_ls_prior_no_prune.pkl"
OUTPUT_FP = (
    "output/fit_penalized_models_revision_full_scipy_ls_prior_no_prune_clamp_removed.pkl"
)


# max_calls=1, max_retries=5: same rationale as penalized_optimization's own
# worker (see model_search.py) -- long sequential GPflow/TF refits in one
# process can accumulate an unbounded resource leak. Errors are caught and
# reported per-metabolite (keeping the stale detail for that one) rather
# than letting one bad component abort the whole batch.
@ray.remote(max_calls=1, max_retries=5)
def refresh_importance_remote(name, model, X_shared, y_np):
    try:
        data = convert_data_to_tensors(X_shared, y_np.reshape(-1, 1))
        model.get_feature_importances(data=data, refit=True)
        return name, model, None
    except Exception as e:
        return name, model, str(e)


def main():
    print("Loading existing fit...")
    with open(INPUT_FP, "rb") as f:
        gps = pickle.load(f)

    X_np = gps.X.to_numpy()
    names = list(gps.models.keys())
    n_metabolites = len(names)

    try:
        ray.init(include_dashboard=False, configure_logging=False)
    except RuntimeError:
        ray.shutdown()
        ray.init(include_dashboard=False, configure_logging=False)

    X_ref = ray.put(X_np)

    print(f"Refreshing {n_metabolites} models...")
    t0 = time.time()
    futures = [
        refresh_importance_remote.remote(
            name, gps.models[name], X_ref, gps.Y[name].to_numpy()
        )
        for name in names
    ]

    # Poll in batches so progress is visible without a separate progress-bar
    # actor (simpler, and nothing async-and-print-order-sensitive to debug
    # if something looks off during a run this long).
    remaining = futures
    n_done = 0
    n_failed = 0
    results = []
    while remaining:
        done, remaining = ray.wait(remaining, num_returns=min(50, len(remaining)))
        for name, model, err in ray.get(done):
            results.append((name, model))
            n_done += 1
            if err is not None:
                n_failed += 1
                print(f"  [{name}] FAILED, keeping stale detail: {err}")
        elapsed = time.time() - t0
        print(
            f"  {n_done}/{n_metabolites} done "
            f"({elapsed/60:.1f} min elapsed, {elapsed/n_done:.2f} sec/metabolite)"
        )

    elapsed = time.time() - t0
    print(
        f"Refreshed {n_metabolites} models in {elapsed/60:.1f} minutes "
        f"({n_failed} failed, kept stale detail for those)"
    )

    for name, m in results:
        gps.models[name] = m

    with open(OUTPUT_FP, "wb") as f:
        pickle.dump(gps, f)
    print(f"Saved to {OUTPUT_FP}")

    ray.shutdown()


if __name__ == "__main__":
    main()
