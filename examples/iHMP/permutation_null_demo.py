"""Small end-to-end demo of the permutation null for component significance.

Design under test (frozen decisions 4-6): the kernel structure is FIXED and
identical for all 564 metabolites, so a null draw costs one fixed-structure
refit plus one drop-one refit -- not a full kernel search. For a covariate
that varies within subject, H0 is imposed by a within-unit CIRCULAR SHIFT of
that covariate along each subject's own time axis: it preserves each
subject's multiset of values and their within-subject autocorrelation, while
breaking alignment with the outcome.

Crucially, the observed statistic is recomputed through the IDENTICAL code
path (shift offset = 0 for every subject) rather than read from the stored
fit. Any optimizer/ELBO idiosyncrasy is then common to both sides and
cancels -- which is the whole reason to prefer this over an assumed-shape
empirical null.

Demo scope: a handful of metabolites x B permutations, targeting the two
hbi components (SE[hbi] = kernel index 5, lin[hbi] = index 6). Writes
output/permutation_null_demo.png.
"""
import pickle
import time

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import ray
from gpflow.utilities import set_trainable

from waveome.utilities import convert_data_to_tensors

INPUT_FP = (
    "output/fit_penalized_models_revision_full_scipy_ls_prior_no_prune_clamp_removed.pkl"
)
SEED = 9102
B = 25                      # permutations per metabolite
TARGETS = {5: "SE[hbi]", 6: "lin[hbi]"}
COV_IDX = 2                 # hbi
UNIT_IDX = 0                # participant_id
TIME_IDX = 5                # study_days -- defines each subject's time order
DEMO = [
    "HILn_QI112",   # strong lin:hbi (stored log_bf 25.3)
    "C18n_QI41",    # strong lin:hbi (18.0)
    "HILp_QI2874",  # strong SE:hbi  (36.6)
    "C8p_QI140",    # null, sits at the floor anchor (-1.0)
    "HILp_QI10330",  # null, below the anchor (-3.0)
]

C_NULL = "#2563eb"
C_OBS = "#d97706"
C_BAR = "#64748b"


def circular_shift(X, col, rng):
    """Rotate `col` within each subject along that subject's time order."""
    out = X.copy()
    for u in np.unique(X[:, UNIT_IDX]):
        idx = np.where(X[:, UNIT_IDX] == u)[0]
        idx = idx[np.argsort(X[idx, TIME_IDX])]
        k = int(rng.integers(0, len(idx)))
        out[idx, col] = X[idx, col][np.roll(np.arange(len(idx)), k)]
    return out


def _bic(m, data):
    # optimize_params(adam/gradient) leaves q_mu/q_sqrt untrainable, which
    # would skew calc_metric's k; restore, then put the flags back.
    a, b = m.q_mu.trainable, m.q_sqrt.trainable
    set_trainable(m.q_mu, True)
    set_trainable(m.q_sqrt, True)
    try:
        return m.calc_metric(data=data, metric="BIC")
    finally:
        set_trainable(m.q_mu, a)
        set_trainable(m.q_sqrt, b)


@ray.remote(max_calls=1, max_retries=5)
def one_draw(model, X, y, shift_seed):
    """One null draw (or the observed statistic when shift_seed is None).

    Returns {kernel_index: log_bf}. Mirrors
    calc_feature_importance_components exactly: refit the full model, then
    warm-start each drop-one refit from that refit.
    """
    Xp = X if shift_seed is None else circular_shift(
        X, COV_IDX, np.random.default_rng(shift_seed)
    )
    data = convert_data_to_tensors(Xp, y.reshape(-1, 1))
    opt = getattr(model, "optimizer", None) or "scipy"

    full = gpflow.utilities.deepcopy(model)
    full.num_trainable_params = np.nan
    full.optimize_params(data=data, optimizer=opt)
    bic_full = _bic(full, data)

    out = {}
    for idx in TARGETS:
        red = gpflow.utilities.deepcopy(full)
        red.kernel.kernels.pop(idx)
        red.num_trainable_params = np.nan
        red.optimize_params(data=data, optimizer=opt)
        out[idx] = float(-0.5 * (bic_full - _bic(red, data)))
    return out


def main():
    with open(INPUT_FP, "rb") as f:
        gps = pickle.load(f)
    X = gps.X.to_numpy()

    ray.init(include_dashboard=False, configure_logging=False)
    X_ref = ray.put(X)

    jobs = []   # (metabolite, is_observed, future)
    for name in DEMO:
        m_ref = ray.put(gps.models[name])
        y_ref = ray.put(gps.Y[name].to_numpy())
        jobs.append((name, True, one_draw.remote(m_ref, X_ref, y_ref, None)))
        for b in range(B):
            jobs.append((name, False, one_draw.remote(
                m_ref, X_ref, y_ref, SEED + 1000 * DEMO.index(name) + b)))

    print(f"{len(jobs)} fits queued ({len(DEMO)} metabolites x (1 observed + "
          f"{B} permutations))")
    t0 = time.time()
    results = ray.get([j[2] for j in jobs])
    print(f"done in {(time.time()-t0)/60:.1f} min")

    obs = {n: {} for n in DEMO}
    null = {n: {i: [] for i in TARGETS} for n in DEMO}
    for (name, is_obs, _), res in zip(jobs, results):
        for idx, v in res.items():
            if is_obs:
                obs[name][idx] = v
            else:
                null[name][idx].append(v)

    # Empirical p against the POOLED null for that stratum (frozen decision
    # 4: p = (1 + #{null >= obs}) / (1 + B)).
    print(f"\n{'metabolite':14s}{'component':10s}{'stored':>9s}{'observed':>10s}"
          f"{'pooled p':>10s}")
    pooled = {i: np.array([v for n in DEMO for v in null[n][i]]) for i in TARGETS}
    for name in DEMO:
        stored = gps.models[name].feature_importance_detail
        for idx, lab in TARGETS.items():
            o = obs[name][idx]
            pool = pooled[idx]
            p = (1 + np.sum(pool >= o)) / (1 + len(pool))
            print(f"{name:14s}{lab:10s}{stored[idx]['log_bf']:9.1f}{o:10.2f}{p:10.4f}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (idx, lab) in zip(axes[:2], TARGETS.items()):
        pool = pooled[idx]
        ax.hist(pool, bins=25, color=C_BAR, alpha=0.6,
                label=f"pooled permutation null (n={len(pool)})")
        for name in DEMO:
            o = obs[name][idx]
            ax.axvline(o, color=C_OBS, lw=2)
            ax.text(o, ax.get_ylim()[1] * 0.92, f" {name}", rotation=90,
                    fontsize=6.5, color=C_OBS, va="top")
        ax.axvline(np.nan, color=C_OBS, lw=2, label="observed (identity shift)")
        ax.set_title(f"{lab}: observed vs permutation null", fontsize=10)
        ax.set_xlabel("log_bf", fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.15, lw=0.5)

    # Exchangeability check: is the null the same across metabolites?
    ax = axes[2]
    data = [null[n][6] for n in DEMO]
    bp = ax.boxplot(data, vert=True, widths=0.6, patch_artist=True,
                    labels=[n[:11] for n in DEMO])
    for patch in bp["boxes"]:
        patch.set_facecolor(C_BAR)
        patch.set_alpha(0.5)
    for med in bp["medians"]:
        med.set_color(C_NULL)
        med.set_linewidth(2)
    ax.set_title("lin[hbi] null per metabolite\n(pooling assumes these agree)",
                 fontsize=10)
    ax.set_ylabel("log_bf under H0", fontsize=9)
    ax.tick_params(axis="x", labelsize=6.5, rotation=30)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.15, lw=0.5, axis="y")

    fig.suptitle("Permutation null demo -- within-unit circular shift of hbi",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("output/permutation_null_demo.png", dpi=130)
    print("\nwrote output/permutation_null_demo.png")
    ray.shutdown()


if __name__ == "__main__":
    main()
