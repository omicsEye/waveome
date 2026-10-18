"""The kernel search scores candidates with the same BIC as the penalized path.

kernel_test (the search's scoring) used to compute BIC from
log_posterior_density, i.e. ELBO + log prior. Search fits carry no horseshoe
prior, but every model carries the LogNormal lengthscale prior, so a kernel
with a lengthscale was charged -2 log p(lengthscale) on top of BIC's own
k*log(n) term -- 4.5 BIC units at a lengthscale of 1, 10.5 at 0.5 -- biasing
the search against exactly the time kernels the simulation plants.
calc_metric (the penalized path) uses the ELBO alone (FINDINGS 21-22).

No pytest in this environment, so this runs directly:

    python tests/test_search_bic.py
"""
import os
import sys

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gpflow  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from waveome.model_search import kernel_test  # noqa: E402
from waveome.utilities import convert_data_to_tensors  # noqa: E402

SEED = 9102


def main():
    rng = np.random.default_rng(SEED)
    t = np.linspace(-2, 2, 60)
    X = t.reshape(-1, 1)
    Y = (np.sin(3 * t) + rng.normal(0, 0.3, len(t))).reshape(-1, 1)

    m, bic = kernel_test(X, Y, gpflow.kernels.SquaredExponential(),
                         likelihood="gaussian", num_restart=1)
    data = convert_data_to_tensors(X, Y)
    ell = float(m.kernel.lengthscales.numpy())
    prior = float(m.log_prior_density().numpy())
    expected = round(float(m.calc_metric(data=data, metric="BIC")), 2)
    print(f"lengthscale {ell:.3f}, log prior {prior:.3f}; "
          f"search BIC {bic}, calc_metric BIC {expected}")

    assert abs(prior) > 0.1, "test needs a non-negligible lengthscale prior term"
    assert bic == expected, \
        f"search BIC {bic} != calc_metric BIC {expected} (prior counted twice?)"
    print("\n[ok] search BIC matches calc_metric")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
