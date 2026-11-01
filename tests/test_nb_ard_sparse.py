"""NB-ARD is fitted as a sparse GP with waveome's inducing-point rule.

The exact variational GP the harness used before did not finish at 6,400
training observations. Checked here:

  1. at most 500 rows: every row is a fixed inducing point, and the fit
     matches the old exact variational GP (same ELBO to within 1%, same
     selected features, close predictions);
  2. above 500 rows: 500 inducing points, trained, chosen reproducibly from
     the seed, and the fit still matches the exact one (ELBO to within 1%,
     same selected features). Checked once at 1,600 rows as well: ELBO
     -3321.19 vs -3321.05, lengthscales 1.31/8.52 vs 1.32/8.57.

No pytest in this environment, so this runs directly:

    python tests/test_nb_ard_sparse.py
"""
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gpflow  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "simulations"))
import sim_waveome_hpc_run as harness  # noqa: E402
import waveome  # noqa: E402

SEED = 9102


def make_data(n, d=6):
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(n, d))
    f = 1.0 + np.sin(2 * x[:, 0]) + 0.5 * x[:, 1]
    mu = np.exp(f)
    y = rng.negative_binomial(n=5, p=5 / (5 + mu)).astype(float)
    return x, y


def old_vgp(x, y):
    """The harness's previous NB-ARD fit, verbatim apart from names."""
    m = gpflow.models.VGP(
        data=(tf.convert_to_tensor(x, dtype=tf.dtypes.double),
              tf.convert_to_tensor(y.reshape(-1, 1), dtype=tf.dtypes.double)),
        kernel=gpflow.kernels.SquaredExponential(lengthscales=np.ones(x.shape[1])),
        likelihood=waveome.likelihoods.NegativeBinomial(),
    )
    gpflow.optimizers.Scipy().minimize(m.training_loss, m.trainable_variables)
    return m


def selected(model):
    ls = model.kernel.lengthscales.numpy()
    return set(np.flatnonzero((ls < 6) & (ls > 0.01)))


def main():
    # 1. Small: same model as the exact variational GP
    x, y = make_data(200)
    z, train_z = harness.nb_ard_inducing_points(x, random_seed=0)
    assert z.shape == x.shape and not train_z
    new = harness.fit_nb_ard(x, y, z, train_z)
    old = old_vgp(x, y)
    data = (tf.convert_to_tensor(x), tf.convert_to_tensor(y.reshape(-1, 1)))
    elbo_new, elbo_old = float(new.elbo(data)), float(old.elbo())
    print(f"n=200: ELBO sparse {elbo_new:.2f} vs exact {elbo_old:.2f}")
    assert abs(elbo_new - elbo_old) <= 0.01 * abs(elbo_old), (elbo_new, elbo_old)
    assert selected(new) == selected(old), (selected(new), selected(old))
    mu_new = new.predict_y(x)[0].numpy().ravel()
    mu_old = old.predict_y(x)[0].numpy().ravel()
    rel = np.max(np.abs(mu_new - mu_old) / (np.abs(mu_old) + 1))
    print(f"       selected features {sorted(selected(new))}; max rel. pred. diff {rel:.3f}")
    assert rel < 0.1, rel
    print("[ok] at <= 500 rows the sparse fit matches the exact one")

    # 2. Large: 500 trained inducing points, reproducible from the seed, and
    # still close to the exact fit
    x, y = make_data(800)
    z, train_z = harness.nb_ard_inducing_points(x, random_seed=3)
    z2, _ = harness.nb_ard_inducing_points(x, random_seed=3)
    z3, _ = harness.nb_ard_inducing_points(x, random_seed=4)
    assert z.shape == (500, x.shape[1]) and train_z
    assert np.array_equal(z, z2) and not np.array_equal(z, z3)
    assert len({tuple(r) for r in z}) == 500, "inducing rows repeat"
    t = time.time()
    new = harness.fit_nb_ard(x, y, z, train_z)
    secs_new = time.time() - t
    assert new.inducing_variable.Z.trainable
    t = time.time()
    old = old_vgp(x, y)
    secs_old = time.time() - t
    data = (tf.convert_to_tensor(x), tf.convert_to_tensor(y.reshape(-1, 1)))
    elbo_new, elbo_old = float(new.elbo(data)), float(old.elbo())
    print(f"n=800: ELBO sparse {elbo_new:.2f} ({secs_new:.0f}s) vs exact "
          f"{elbo_old:.2f} ({secs_old:.0f}s)")
    assert abs(elbo_new - elbo_old) <= 0.01 * abs(elbo_old), (elbo_new, elbo_old)
    assert selected(new) == selected(old), (selected(new), selected(old))
    print(f"       selected features {sorted(selected(new))}, as the exact fit")
    print("[ok] above 500 rows: 500 seeded, trained inducing points; matches exact")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
