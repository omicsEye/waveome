"""Non-finite permutation draws fail like draws that raise (TODO A2).

A refit whose BIC is NaN gives a NaN log_bf without raising. Before the fix
it was saved as an ordinary draw, the quantile regression in
calc_permutation_pvalues then refused it ("exog contains inf or nans") after
every draw had finished, and a resumed run treated the NaN rows as torn lines
and appended them again. Checked here, with a model whose BIC is forced to
NaN:

  1. NaN null draws (one outcome, some permutations): the run finishes, the
     draws kept are exactly an unforced run's draws minus the failed ones,
     and nothing non-finite reaches the draws, the checkpoint or the p-values;
  2. NaN observed statistic (another outcome, every refit): a warning names
     the untestable (outcome, covariate) pairs and they get no p-value;
  3. resuming from that checkpoint re-queues the failed draws, which fail
     again, and adds no rows.

No pytest in this environment, so this runs directly:

    python tests/test_permutation_nonfinite.py
"""
import os
import shutil
import sys
import tempfile
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from test_permutation_resume import SEED, build_fit  # noqa: E402

COVS = ["cov_a", "cov_b"]
KEY = ["metabolite", "covariate", "kernel_idx", "draw"]


def force_nan_bic(model, col, orig, thr, always=False):
    """Make `model` report a NaN BIC when column `col` is permuted and its
    first value exceeds `thr` (or for every refit, if `always`)."""
    base = type(model)

    class NaNBIC(base):
        def calc_metric(self, data=None, metric="BIC"):
            x = np.asarray(data[0])[:, col]
            if always or (not np.allclose(x, orig) and x[0] > thr):
                return np.nan
            return base.calc_metric(self, data=data, metric=metric)

    model.__class__ = NaNBIC


def run(gps, ck):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = gps.permutation_significance(
            covariates=COVS, B0=6, B1=6, random_seed=SEED,
            checkpoint_path=ck, verbose=False)
    msgs = [str(w.message) for w in caught if "no observed statistic" in str(w.message)]
    return res, gps.permutation_draws, msgs


def main():
    tmp = tempfile.mkdtemp()
    try:
        # Reference: no forced failures
        ref_res, ref, _ = run(build_fit(), os.path.join(tmp, "ref.csv"))

        gps = build_fit()
        col = gps.feat_names.index("cov_b")
        orig = gps.X.to_numpy()[:, col].astype(float)
        unit0 = orig[gps.X["id"].to_numpy() == gps.X["id"].iloc[0]]
        force_nan_bic(gps.models["y1"], col, orig, thr=np.median(unit0))
        force_nan_bic(gps.models["y2"], col, orig, thr=0, always=True)
        ck = os.path.join(tmp, "nan.csv")
        res, draws, msgs = run(gps, ck)

        # 1. NaN null draws: dropped whole, nothing else changes
        assert np.isfinite(draws.log_bf).all(), "non-finite draw kept"
        disk = pd.read_csv(ck)
        assert np.isfinite(disk.log_bf).all(), "non-finite draw checkpointed"
        assert not disk.duplicated(KEY).any(), "duplicate draws on disk"
        lost = (ref.query("metabolite == 'y1' and covariate == 'cov_b' and draw >= 0")
                .draw.nunique()
                - draws.query("metabolite == 'y1' and covariate == 'cov_b' and draw >= 0")
                .draw.nunique())
        assert 0 < lost < 6, f"expected some, not all, y1 cov_b draws to fail ({lost})"
        kept_ref = ref[ref.metabolite != "y2"].merge(
            draws[KEY], on=KEY, how="inner")
        assert len(kept_ref) == len(draws), "kept draws differ from the reference"
        both = kept_ref.merge(draws, on=KEY, suffixes=("_ref", ""))
        assert np.allclose(both.log_bf_ref, both.log_bf), "kept draws changed value"
        y1b = res.query("metabolite == 'y1' and covariate == 'cov_b'")
        assert len(y1b) and y1b.p_value.notna().all(), "y1 cov_b lost its p-values"
        assert res.p_value.notna().all() and res.q_value.notna().all()
        print(f"[1] {lost} of 6 y1/cov_b null draws were NaN: dropped whole; "
              "kept draws identical to an unforced run; p-values computed")

        # 2. NaN observed statistic: warned, untestable, no crash
        assert (draws.metabolite == "y2").sum() == 0
        assert not (res.metabolite == "y2").any(), "y2 should have no p-values"
        assert len(msgs) == 1 and "('y2', 'cov_a')" in msgs[0] \
            and "('y2', 'cov_b')" in msgs[0], f"warning missing or wrong: {msgs}"
        assert set(ref_res.metabolite) - set(res.metabolite) == {"y2"}
        print("[2] y2's observed refit was NaN: warned, no p-value, "
              "other outcomes unaffected")

        # 3. Resume: failed draws re-queued, fail again, nothing appended
        n_disk = len(disk)
        res2, draws2, _ = run(gps, ck)
        disk2 = pd.read_csv(ck)
        assert len(disk2) == n_disk, f"resume appended rows: {n_disk} -> {len(disk2)}"
        assert not disk2.duplicated(KEY).any()
        assert np.allclose(res.sort_values(["metabolite", "covariate", "kernel_type"]).p_value,
                           res2.sort_values(["metabolite", "covariate", "kernel_type"]).p_value)
        print("[3] resume re-ran the failed draws: no rows added, same p-values")
        print("\nall checks passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
