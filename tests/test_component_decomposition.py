"""The plotted kernel components must be an additive decomposition.

`individual_kernel_predictions(marginal=True)` used to isolate a component by
swapping the model's kernel for that sub-kernel and calling `predict_f`. The
variational parameters were fitted against the FULL kernel, so that rebuilt
K_zz from one component and re-interpreted q_mu against the wrong matrix --
the components did not sum to the model and the curve could come out with the
wrong sign (FINDINGS.md section 29).
"""
import gpflow
import numpy as np
import tensorflow as tf

from waveome.utilities import individual_kernel_predictions


def _toy_svgp(seed=9102, n=60, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    # signal with opposite-sign contributions so a sign flip is detectable
    Y = (2.0 * X[:, 0] - 1.5 * X[:, 1]).reshape(-1, 1) + 0.1 * rng.normal(size=(n, 1))
    kernel = gpflow.kernels.Sum([
        gpflow.kernels.Linear(active_dims=[0], variance=1.0),
        gpflow.kernels.Linear(active_dims=[1], variance=1.0),
        gpflow.kernels.SquaredExponential(active_dims=[2], variance=0.5),
    ])
    m = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=X.copy(),
        mean_function=gpflow.mean_functions.Constant(np.array([0.7])),
        whiten=True,
    )
    gpflow.optimizers.Scipy().minimize(
        m.training_loss_closure((X, Y)), m.trainable_variables,
        options={"maxiter": 200},
    )
    return m, X, Y


def test_components_sum_to_full_model():
    """Sum of component means == full model mean, up to the constant mean
    function counted once per component."""
    m, X, Y = _toy_svgp()
    n_k = len(m.kernel.kernels)

    total = np.zeros((X.shape[0], 1))
    for i in range(n_k):
        mu, _, _, _ = individual_kernel_predictions(
            model=m, kernel_idx=i, data=(X, Y), X=X, marginal=True
        )
        total += np.asarray(mu).reshape(-1, 1)

    full = m.predict_f(X)[0].numpy().reshape(-1, 1)
    const = float(np.ravel(m.mean_function.c.numpy())[0])
    # the mean function rides on every component, so it lands n_k times
    recovered = total - (n_k - 1) * const

    np.testing.assert_allclose(recovered, full, atol=1e-6)


def test_component_direction_matches_full_model():
    """With only one covariate varying, the component's slope must agree in
    sign with the full model's -- the failure the old branch produced."""
    m, X, Y = _toy_svgp()
    grid = np.zeros((50, X.shape[1]))
    grid[:, 0] = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)

    full = m.predict_f(grid)[0].numpy().ravel()
    mu, _, _, _ = individual_kernel_predictions(
        model=m, kernel_idx=0, data=(X, Y), X=grid, marginal=True
    )
    comp = np.asarray(mu).ravel()

    assert np.sign(full[-1] - full[0]) == np.sign(comp[-1] - comp[0])
    # dim 0 has a positive coefficient in the generating process
    assert comp[-1] > comp[0]


def test_variance_is_positive():
    m, X, Y = _toy_svgp()
    _, var, _, _ = individual_kernel_predictions(
        model=m, kernel_idx=0, data=(X, Y), X=X, marginal=True
    )
    assert np.all(np.asarray(var) > -1e-8)
