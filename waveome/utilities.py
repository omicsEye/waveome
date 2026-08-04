import contextlib
import time
import warnings
from ctypes import ArgumentError
from xml.etree.ElementInclude import include

import gpflow
import joblib
import numpy as np
import psutil
import ray
import scipy
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
from gpflow.utilities import set_trainable
from joblib import Parallel, delayed
from ray.experimental import tqdm_ray
from scipy.stats import norm
from tensorflow_probability import distributions as tfd

from .kernels import Empty
from .likelihoods import (
    NegativeBinomial,
    ZeroInflatedNegativeBinomial,
    negative_binomial,
)

# from multiprocessing import Value


f64 = gpflow.utilities.to_default_float

# Numerical pre-filter for kernel-component pruning (cut_kernel_components in
# model_classes.py and regularization.py) -- not a significance/selection rule.
VAR_CUTOFF_DEFAULT = 1e-8

# calc_feature_importance_components: value used to evaluate a component's
# counterfactual "removed" state without refitting, for components already
# below VAR_CUTOFF_DEFAULT (see _clamp_result). Far more extreme than any
# naturally-collapsed fitted variance observed (~1e-20 to ~1e-90), but not so
# extreme it destabilizes the Cholesky decomposition in log_posterior_density.
COMPONENT_CLAMP_VALUE = 1e-150


def set_precision(precision: str = "float64"):
    """
    Set the default floating point precision for the library.

    Parameters
    ----------
    precision : str
        Either "float32" or "float64".
    """
    if precision not in ["float32", "float64"]:
        raise ValueError("Precision must be either 'float32' or 'float64'")

    if precision == "float32":
        gpflow.config.set_default_float(np.float32)
    else:
        gpflow.config.set_default_float(np.float64)


def get_precision():
    """
    Get the current default floating point precision.
    """
    dtype = gpflow.config.default_float()
    if dtype == np.float32:
        return "float32"
    else:
        return "float64"


# Set default precision to float64
set_precision("float64")


def convert_data_to_tensors(X: np.array, Y: np.array):

    tensor_tuple = (
        tf.convert_to_tensor(X, dtype=gpflow.default_float()),
        tf.convert_to_tensor(Y, dtype=gpflow.default_float()),
    )

    return tensor_tuple


def calc_bic(loglik: float, n: int, k: int):
    """Returns the Bayesian Information Criteria (BIC) for log likelihood.

    Parameters
    ---------
    loglik: float
        Log-likelihood of observations under model.
    n: int
        Number of observations.
    k: int
        Number of traininable parameters.

    Returns
    -------
    float
        BIC
    """
    return k * np.log(n) - 2 * loglik


def coregion_freeze(k):
    """Freeze parameters associated with coregion kernel,
    for individual level effets.

    Parameters
    ----------
    k: gpflow.kernel.Kernel
        Model's kernel which should include a coregion.

    Returns
    -------
    """

    if k.name == "coregion":
        # print('Found coregion kernel, freezing parameters.')
        k.W.assign(np.zeros_like(k.W))
        k.kappa.assign(np.ones_like(k.kappa))
        set_trainable(k.W, False)
        set_trainable(k.kappa, False)

    return None


def coregion_search(kern_list):
    """Search through GP kernel list to find coregion kernels."""

    for k in kern_list:
        if hasattr(k, "kernels"):
            coregion_search(k.kernels)
        else:
            coregion_freeze(k)


def calc_rsquare(m, data=None):
    """
    Calculate the r-squared values of each kernel component.
    """

    # Save output list
    rsq = []

    if data is None:
        # Pull off data from stored model
        X = m.data[0].numpy()
        Y = m.data[1].numpy()
    else:
        X = data[0] if isinstance(data[0], np.ndarray) else data[0].numpy()
        Y = data[1] if isinstance(data[1], np.ndarray) else data[1].numpy()

    # Make copy of model
    m_copy = gpflow.utilities.deepcopy(m)

    # Calculate the mean of the outcome
    Y_bar = Y.mean()

    # Calculate sum of squares error
    sse = np.sum((Y - Y_bar) ** 2)

    # Calculate overall model predictions
    mu_all_hat, var_all_hat = m.predict_y(X)
    ssr_total = np.sum((Y - mu_all_hat) ** 2)
    # print(f"{sse=}, {ssr_total=}")
    total_rsq = 1 - (ssr_total / sse)

    # For each kernel component gather predictions
    ssr_list = []
    k = m.kernel
    if k.name == "sum":
        for k_idx in range(len(k.kernels)):
            # Break off kernel component
            # m_copy.kernel = k_sub
            # mu_hat, var_hat = m_copy.predict_y(X)
            mu_hat, var_hat, samps_, cov_hat = individual_kernel_predictions(
                model=m, kernel_idx=k_idx, data=(X, Y), X=X
            )
            ssr_list += [np.sum((mu_all_hat - mu_hat) ** 2)]

        for k_idx in range(len(k.kernels)):
            rsq += [
                np.round(total_rsq * (1 - ssr_list[k_idx] / sum(ssr_list)), 3)
            ]
    else:
        mu_hat, var_hat = m_copy.predict_y(X)
        # ssr = np.sum((mu_all_hat - mu_hat) ** 2)
        rsq += [np.round(total_rsq, 3)]

    # Gather the final bit for noise
    # rsq += [np.round(1 - sum(rsq),3)]
    rsq += [np.round(1 - total_rsq, 3)]

    return rsq


def calc_residuals(m, X=None, Y=None, resid_type="raw"):
    """
    Calculate pearson residuals from model
    """
    # Set x values if none given
    if X is None:
        X = m.data[0]
    # Same for y values
    if Y is None:
        Y = m.data[1]
    # Get observed predictions and variance
    # mean_resp, var_resp = m.predict_y(m.data[0])
    mean, var = m.predict_f(X)
    mean_resp = m.likelihood._conditional_mean(X=X, F=mean)
    var_resp = m.likelihood._conditional_variance(X=X, F=mean)

    if resid_type == "raw":
        resids = (tf.cast(Y, gpflow.default_float()) - mean_resp).numpy()
    elif resid_type == "deviance":
        null_resids, resids = calc_deviance_explained(
            model=m,
            data=(X, Y),
            model_mu=mean_resp,
            model_var=var_resp,
            aggregate=False,
            return_deviance_explained=False,
        )
    elif resid_type == "pearson":
        # Calculate standardized residuals
        resids = (
            (tf.cast(Y, gpflow.default_float()) - mean_resp)
            / np.sqrt(var_resp)
        ).numpy()
    else:
        raise ArgumentError("resid_type can only be 'raw' or pearson'")

    return resids


def calc_bhattacharyya_dist(model1, model2, X):
    """
    Calculate the Bhattacharyya distance between two resulting
    MVNormal distributions.
    """

    # Calculate means and variances
    mu1, var1 = model1.predict_f(X)
    mu2, var2 = model2.predict_f(X)

    # Also calculate covariance matrices
    # Pull kernel covariance matrices
    cov1 = model1.kernel.K(X)
    cov2 = model2.kernel.K(X)

    # Then add likelihood noise if necessary
    if model1.name == "gpr" and model2.name == "gpr":
        cov1 += tf.linalg.diag(
            tf.repeat(model1.likelihood.variance, X.shape[0])
        )
        cov2 += tf.linalg.diag(
            tf.repeat(model2.likelihood.variance, X.shape[0])
        )

    # Calculate average sigma
    cov_all = (cov1 + cov2) / 2.0

    # After that calculate closed form of Bhattacharyya distance
    dist_b = 0.5 * np.log(
        tf.linalg.det(cov_all)
        / (np.sqrt(tf.linalg.det(cov1) * tf.linalg.det(cov2)))
    )

    return dist_b


def replace_kernel_variables(k_name, col_names):
    """
    Takes in indexed kernel names and original column names, then replaces
    and spits out new string.
    """

    # Make copy of kernel name
    new_k_name = k_name

    for i, c in enumerate(col_names):
        new_k_name = new_k_name.replace("[" + str(i) + "]", "[" + c + "]")

    return new_k_name


def check_if_model_exists(model_name, model_list):
    """
    Checks if current model name is in list of fit models.
    """
    found_model = None

    # First split models into additive components
    model_name_split = model_name.split("+")
    model_list_split = [x.split("+") for x in model_list]

    # Then order the resulting product pieces
    model_name_split_ordered = ["".join(sorted(x)) for x in model_name_split]
    # model_list_split_ordered = [
    #     "".join(sorted(x)) for y in model_list_split for x in y
    # ]

    term_diff = [
        set(model_name_split_ordered) ^ set(["".join(sorted(x)) for x in y])
        for y in model_list_split
    ]

    if set() in term_diff:
        found_model = True
    else:
        found_model = False

    return found_model


def hmc_sampling(
    model,
    burn_in=500,
    samples=1000,
    random_seed=None,
    step_size=0.01,
    accept_prob=0.9,
    num_adaptation_steps=100,
):
    model = gpflow.utilities.deepcopy(model)

    # Set priors if they don't already have them
    for p in model.parameters:
        if p.prior is None:
            p.prior = tfd.Gamma(f64(2), f64(2))

    # Set helper
    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters
    )

    # Set HMC options
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn,
        num_leapfrog_steps=10,
        step_size=step_size,
    )
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc,
        num_adaptation_steps=num_adaptation_steps,
        target_accept_prob=f64(accept_prob),
        adaptation_rate=0.1,
    )

    # Run sampler
    samples, traces = tfp.mcmc.sample_chain(
        # num_results=ci_niter(samples),
        # num_burnin_steps=ci_niter(burn_in),
        num_results=samples,
        num_burnin_steps=burn_in,
        current_state=hmc_helper.current_state,
        kernel=adaptive_hmc,
        seed=random_seed,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )

    # Get constrained values
    strain_samples = hmc_helper.convert_to_constrained_values(samples)

    return {
        "samples": strain_samples,
        "unconstrained_samples": samples,
        "traces": traces,
    }


def print_kernel_names(kernel, with_idx=False):
    names = []

    if kernel is None:
        return ""

    if hasattr(kernel, "kernels") is False:
        if with_idx:
            return kernel.name + "[" + str(kernel.active_dims[0]) + "]"
        else:
            return kernel.name
    elif kernel.name == "sum":
        return [print_kernel_names(x, with_idx) for x in kernel.kernels]
    elif kernel.name == "product":
        return "*".join(
            [print_kernel_names(x, with_idx) for x in kernel.kernels]
        )
    return names


# def adam_opt_params(m, iterations=500, eps=0.1):
#     prev_loss = np.Inf
#     for i in range(iterations):
#         tf.optimizers.Adam(learning_rate=0.1, epsilon=0.1).minimize(
#             m.training_loss, m.trainable_variables
#         )

#         if abs(prev_loss - m.training_loss()) < eps:
#             break
#         else:
#             prev_loss = m.training_loss()

#         if i % 50 == 0:
#             print(f'Current loss: {prev_loss}')
#     return None


def variance_contributions(m, k_names, lik="gaussian"):
    """
    Takes a GP model and returns the percent of variance explained for each
    additive component.
    """

    variance_list = []

    # Split kernel into additive pieces
    kernel_names = k_names.split("+")

    # Check if there is only one kernel component, otherwise go through all
    if len(kernel_names) == 1:
        if m.kernel.name == "product":
            prod_var = 1
            for k in m.kernel.kernels:
                if k.name == "periodic":
                    prod_var *= k.base_kernel.variance.numpy().round(3)
                else:
                    prod_var *= k.variance.numpy().round(3)
            variance_list += [prod_var.tolist()]

        elif m.kernel.name == "sum":
            sum_var = 0
            for k in m.kernel.kernels:
                if k.name == "periodic":
                    sum_var += k.base_kernel.variance.numpy().round(3)
                else:
                    sum_var += k.variance.numpy().round(3)
            variance_list += [sum_var.tolist()]

        elif m.kernel.name == "periodic":
            variance_list += [m.kernel.base_kernel.variance.numpy().round(3)]
        elif m.kernel.name == "empty":
            variance_list += [0.0]
        else:
            variance_list += [m.kernel.variance.numpy().round(3)]
    else:
        for k in range(len(kernel_names)):
            if m.kernel.kernels[k].name == "product":
                prod_var = 1
                for k2 in m.kernel.kernels[k].kernels:
                    if k2.name == "periodic":
                        prod_var *= k2.base_kernel.variance.numpy().round(3)
                    else:
                        prod_var *= k2.variance.numpy().round(3)
                variance_list += [prod_var.tolist()]

            elif m.kernel.kernels[k].name == "sum":
                sum_var = 0
                for k2 in m.kernel.kernels[k].kernels:
                    if k2.name == "periodic":
                        sum_var += k2.base_kernel.variance.numpy().round(3)
                    else:
                        sum_var += k2.variance.numpy().round(3)
                variance_list += [sum_var.tolist()]

            elif m.kernel.kernels[k].name == "periodic":
                variance_list += [
                    m.kernel.kernels[k]
                    .base_kernel.variance.numpy()
                    .round(3)
                    .tolist()
                ]

            else:
                variance_list += [
                    m.kernel.kernels[k].variance.numpy().round(3).tolist()
                ]

    # Get likelihood variance
    if lik == "gaussian":
        variance_list += [m.likelihood.variance.numpy().round(3).tolist()]
    else:
        variance_list += [np.std(calc_residuals(m)) ** 2]
    #     elif lik == 'exponential':
    #     elif lik == 'poisson':
    #     elif lik == 'gamma':
    #     elif lik == 'bernoulli':
    #         variance_list +=
    #     else:
    #         raise ValueError('Unknown likelihood function specified.')
    return variance_list


def variance_contributions_diag(m, lik="gaussian"):
    # TODO: Add arg for data and pass into individual_kernel_predictions()
    variance_list = []
    k = m.kernel

    # Extract variance from kernel components
    if k.name == "sum":
        for i in range(len(k.kernels)):
            mu_, var_, samps_, cov_ = individual_kernel_predictions(
                model=m, kernel_idx=i, X=m.data[0]
            )
        # for k_sub in k.kernels:
        # variance_list += [np.mean(k_sub.K_diag(m.data[0]))]
    elif k.name == "product":
        temp_prod = np.ones_like(m.data[0][:, 0])
        for k_sub in k.kernels:
            temp_prod *= k_sub.K_diag(m.data[0])
        variance_list += [np.mean(temp_prod)]
    else:
        variance_list += [np.mean(k.K_diag(m.data[0]))]

    # Extract variance from likelihood function
    if lik == "gaussian":
        variance_list += [m.likelihood.variance.numpy().round(3).tolist()]
    else:
        variance_list += [np.std(calc_residuals(m)) ** 2]
    return variance_list


def calc_deviance_explained(
    model,
    data=None,
    model_mu=None,
    model_var=None,
    base_mu=None,
    base_var=None,
    aggregate=True,
    return_deviance_explained=True,
    return_loglik=False,
):
    """Calculates explained deviance for model components compared to null
    model.
    """

    # Get likelihood of model
    lk_fn = model.likelihood.name

    # Retrieve likelihood function
    try:
        log_dens_fn = getattr(gpflow.logdensities, lk_fn)
    except AttributeError:
        log_dens_fn = getattr(
            scipy.stats, "nbinom" if lk_fn == "negative_binomial" else lk_fn
        ).logpmf

    # Calculate log likelihoods
    if lk_fn == "gaussian":
        y_var = np.var(data[1])
        sat_ll = log_dens_fn(x=data[1], mu=data[1], var=y_var)
        base_ll = log_dens_fn(
            x=data[1],
            mu=np.mean(data[1]) if base_mu is None else base_mu,
            var=y_var,
        )
        mod_ll = log_dens_fn(x=data[1], mu=model_mu, var=y_var)
    elif lk_fn in ["bernoulli", "poisson"]:
        sat_ll = log_dens_fn(data[1], data[1])
        base_ll = log_dens_fn(
            data[1], np.mean(data[1]) if base_mu is None else base_mu
        )
        mod_ll = log_dens_fn(data[1], model_mu)
    elif lk_fn == "negative_binomial":

        # sigma^2 = mu + mu^2 * alpha
        # alpha = mu^2 / (sigma^2 - mu)
        try:
            alpha_val = model.likelihood.alpha.numpy()
        except AttributeError:
            alpha_val = 1.0

        sat_mu_ = np.array(data[1]) + 1e-6
        sat_ll = negative_binomial(
            m=sat_mu_, Y=np.array(data[1]), alpha=alpha_val
        )

        base_mu_ = max(1e-6, np.mean(data[1])) if base_mu is None else base_mu
        base_ll = negative_binomial(
            m=base_mu_, Y=np.array(data[1]), alpha=alpha_val
        )

        mod_ll = negative_binomial(
            m=model_mu, Y=np.array(data[1]), alpha=alpha_val
        )

    else:
        raise ValueError("Unknown likelihood to calculate deviance")

    if return_loglik:
        return base_ll, mod_ll, sat_ll

    # Now calculate the deviances
    if aggregate:
        # Null deviance should always be greater than model deviance!
        null_deviance = max(0, 2 * np.sum(sat_ll - base_ll))
        # Clip to zero to avoid negatives
        model_deviance = max(0, 2 * np.sum(sat_ll - mod_ll))
        # Then calculate deviance explained
        deviance_explained = (
            1 - (model_deviance / null_deviance) if null_deviance > 0 else 0
        )
    else:
        null_deviance = np.clip(2 * (sat_ll - base_ll), a_min=0, a_max=np.inf)
        model_deviance = np.clip(2 * (sat_ll - mod_ll), a_min=0, a_max=np.inf)
        deviance_explained = 1 - np.divide(
            model_deviance,
            null_deviance,
            out=np.ones_like(model_deviance, dtype="float"),
            where=(null_deviance != 0),
        )

    if return_deviance_explained:
        return deviance_explained
    else:
        return null_deviance, model_deviance


def _calc_feature_importance_components_legacy(
    model, data=None, return_value="log_bf"
):
    """Legacy (pre-refit) no-refit plug-in predictive-log-density behavior.
    Kept only for `calc_feature_importance_components(..., refit=False)`.
    """

    # Save output list
    de_list = []

    # For each kernel component gather predictions
    k = model.kernel

    # Get full model prediction and deviance
    full_mu_hat, full_var_hat = model.predict_y(data[0])
    null_lls, mod_lls, sat_lls = calc_deviance_explained(
        model=model,
        data=data,
        model_mu=full_mu_hat,
        model_var=full_var_hat,
        return_deviance_explained=False,
        aggregate=False,
        return_loglik=True,
    )

    # calculate model deviance: 1 - (full_dev / null_dev)
    if np.sum(sat_lls) >= np.sum(mod_lls) and np.sum(mod_lls) >= np.sum(
        null_lls
    ):
        full_de = 1 - (
            -2 * np.sum(mod_lls - sat_lls) / (-2 * np.sum(null_lls - sat_lls))
        )
        full_de = max(min(1, full_de), 0)
    else:
        full_de = 0

    if k.name == "sum":
        for k_idx in range(len(k.kernels)):

            # Now get other components predictions (ignoring this component)
            model_copy = gpflow.utilities.deepcopy(model)
            _ = model_copy.kernel.kernels.pop(k_idx)
            mod_mu_hat, mod_var_hat = model_copy.predict_y(data[0])

            # Get likelihood estimate (w/o component)
            null_lls, sub_mod_lls, sat_lls = calc_deviance_explained(
                model=model_copy,
                data=data,
                model_mu=mod_mu_hat,
                model_var=mod_var_hat,
                return_deviance_explained=False,
                aggregate=False,
                return_loglik=True,
            )

            # Calc deviance explained by looking at deviance without feature
            # included ("best" is fit model not saturated)
            if return_value == "statistic":
                scaled_de = np.round(
                    -2 * (np.sum(sub_mod_lls) - np.sum(mod_lls)), 1
                )
                scaled_de = max(scaled_de, 0)
            elif return_value == "log_bf":
                scaled_de = np.round(np.sum(mod_lls) - np.sum(sub_mod_lls), 1)
            else:
                scaled_de = 1 - (
                    -2
                    * np.sum(sub_mod_lls - mod_lls)
                    / (-2 * np.sum(null_lls - mod_lls))
                )
                scaled_de = np.round(max(min(1, scaled_de), 0), 3)

            # Append to list
            de_list += [scaled_de]

    else:
        # If there is just a single term
        if k.name == "constant":
            de_list += [0.0]
        else:
            if return_value == "statistic":
                de_list += [
                    np.round(-2 * (np.sum(null_lls) - np.sum(mod_lls)), 1)
                ]
            elif return_value == "log_bf":
                de_list += [np.round((np.sum(mod_lls) - np.sum(null_lls)), 1)]
            else:
                de_list += [np.round(full_de, 3)]

    # Gather the final bit for leftover noise
    de_list += [np.round(1 - full_de, 3)]

    return de_list


def feature_importance_detail_to_flat(detail_list, return_value="log_bf"):
    """Project a `calc_feature_importance_components(..., full_detail=True)`
    detail list down to the flat scalar-per-component list historically
    returned by `calc_feature_importance_components`/`get_feature_importances`.

    The trailing "leftover noise" entry is always reported in deviance-
    explained units, regardless of `return_value` -- matching the pre-refit
    contract.
    """
    key = {"statistic": "delta_bic", "log_bf": "log_bf"}.get(
        return_value, "deviance_explained"
    )
    flat = [d[key] for d in detail_list[:-1]]
    flat.append(detail_list[-1]["deviance_explained"])
    return flat


def calc_feature_importance_components(
    model,
    data=None,
    return_value="log_bf",
    refit=True,
    refit_options=None,
    full_detail=False,
):
    """Calculate an evidence statistic and marginal deviance explained for
    each additive kernel component, by refitting the model with that
    component dropped (frozen decision: refit required, warm-started from
    the full model's fitted parameters).

    Exception: a component whose fitted variance is already below
    VAR_CUTOFF_DEFAULT skips the refit and instead evaluates the same
    fitted model's likelihood with that component's variance clamped to an
    extreme near-zero value (COMPONENT_CLAMP_VALUE), using k_full - 1 for
    the comparison's parameter count. This is safe specifically because the
    component is already collapsed: the full model's other parameters were
    already optimized as if it contributed nothing, so a refit would barely
    move them -- but it is not used above the floor, where a refit-free
    comparison would be a biased comparison.

    Parameters
    ----------
    return_value: str
        "log_bf" (default): log Bayes factor, log_bf = -0.5 * delta_bic.
        "statistic": raw delta_bic = BIC_full - BIC_reduced. Sign convention:
            more negative delta_bic (equivalently, larger log_bf) = more
            evidence the component matters.
        anything else: marginal (drop-one) deviance explained.
        Ignored when `full_detail=True`.
    refit: bool
        If True (default), each reduced (component-dropped) model is
        re-optimized -- warm-started from the full model's fitted parameters
        via deepcopy -- before delta_bic / deviance explained are computed.
        If False, reproduces the legacy no-refit plug-in predictive-log-
        density behavior (`_calc_feature_importance_components_legacy`).
    refit_options: dict
        Passed through to `model.optimize_params(...)` for each reduced-
        model refit. Ignored when refit=False.
    full_detail: bool
        If True, return a list of dicts (one per component, in the same
        order as the flat list, plus the trailing leftover-noise entry) with
        keys "delta_bic", "log_bf", "deviance_explained" -- all computed
        from the same refit, so callers needing more than one quantity do
        not have to refit twice. Ignored when refit=False.
    """
    if not refit:
        return _calc_feature_importance_components_legacy(
            model, data=data, return_value=return_value
        )

    # Default the refit optimizer to whatever the full model itself was fit
    # with, so delta_bic reflects only the dropped component, not an
    # optimizer-quality mismatch between the two sides of the comparison.
    # Previously hardcoded to 'adam/gradient' regardless of the full
    # model's optimizer -- confirmed via a 7332-component real-data check
    # (fit_penalized_models_revision_full_scipy_no_prune.pkl) that this
    # mismatch only affects ~1.6% of components, but by up to several
    # thousand log_bf units when it does (e.g. one component: 10130.2 with
    # a mismatched adam/gradient refit vs 10.4 once refit with the full
    # model's own scipy optimizer). See docs/revision/FINDINGS.md, "T2
    # (continued)". Caveat carried over from the prior default:
    # scipy/L-BFGS-B can hit a vanishing-gradient trap when a component's
    # variance has already collapsed near zero -- the _needs_clamp branch
    # below routes genuinely-collapsed components away from a refit
    # entirely, which should catch most of that risk, but it's worth
    # watching for if scipy refits start failing to converge.
    default_optimizer = getattr(model, "optimizer", None) or "adam/gradient"
    refit_options = {"optimizer": default_optimizer, **(refit_options or {})}
    actual_optimizer = refit_options["optimizer"]
    if getattr(model, "optimizer", None) not in (None, actual_optimizer):
        warnings.warn(
            f"Full model was fit with optimizer={model.optimizer!r}, but "
            f"component refits use {actual_optimizer!r}; delta_bic may "
            "partly reflect differing optimizer quality, not just the "
            "dropped component."
        )
    k = model.kernel

    def _bic(m):
        # optimize_params(optimizer="adam/gradient") leaves q_mu/q_sqrt
        # untrainable afterward, which would otherwise skew calc_metric's
        # BIC (k = len(trainable_parameters)). Restore before counting,
        # then put the original trainable state back -- m may be the
        # caller's live model, not a deepcopy, so this must not
        # permanently flip its trainable flags as a side effect.
        q_mu_trainable = m.q_mu.trainable
        q_sqrt_trainable = m.q_sqrt.trainable
        set_trainable(m.q_mu, True)
        set_trainable(m.q_sqrt, True)
        try:
            return m.calc_metric(data=data, metric="BIC")
        finally:
            set_trainable(m.q_mu, q_mu_trainable)
            set_trainable(m.q_sqrt, q_sqrt_trainable)

    # Full-model deviance decomposition (used as the fixed reference for
    # marginal deviance explained, and for the leftover-noise entry).
    full_mu_hat, full_var_hat = model.predict_y(data[0])
    null_lls, mod_lls, sat_lls = calc_deviance_explained(
        model=model,
        data=data,
        model_mu=full_mu_hat,
        model_var=full_var_hat,
        return_deviance_explained=False,
        aggregate=False,
        return_loglik=True,
    )
    if np.sum(sat_lls) >= np.sum(mod_lls) and np.sum(mod_lls) >= np.sum(
        null_lls
    ):
        full_de = 1 - (
            -2 * np.sum(mod_lls - sat_lls) / (-2 * np.sum(null_lls - sat_lls))
        )
        full_de = max(min(1, full_de), 0)
    else:
        full_de = 0

    full_bic = _bic(model)
    if not np.isfinite(full_bic):
        raise ValueError(
            "calc_feature_importance_components: the full model's own BIC "
            "is non-finite (fitted model appears degenerate); cannot "
            "compute component importances. Re-fit the full model and "
            "retry -- this has been observed as a rare, non-reproducible "
            "numerical fault, not a deterministic property of the data."
        )
    k_full = len(model.trainable_parameters)

    def _refit(model_copy):
        # Warm start: model_copy already holds the full model's fitted
        # values (from the deepcopy) for every surviving parameter.
        model_copy.num_trainable_params = np.nan
        model_copy.optimize_params(data=data, **refit_options)
        return model_copy

    def _component_result(model_copy):
        reduced_bic = _bic(model_copy)
        delta_bic = full_bic - reduced_bic
        log_bf = -0.5 * delta_bic

        mod_mu_hat, mod_var_hat = model_copy.predict_y(data[0])
        null_lls_r, sub_mod_lls, _ = calc_deviance_explained(
            model=model_copy,
            data=data,
            model_mu=mod_mu_hat,
            model_var=mod_var_hat,
            return_deviance_explained=False,
            aggregate=False,
            return_loglik=True,
        )
        # Fraction of the full model's gain-over-null attributable to this
        # component (high = important, ~0 = null).
        denom = -2 * np.sum(null_lls_r - mod_lls)
        if denom != 0:
            marginal_de = (-2 * np.sum(sub_mod_lls - mod_lls)) / denom
            marginal_de = np.round(max(min(1, marginal_de), 0), 3)
        else:
            marginal_de = 0.0

        return {
            "delta_bic": np.round(delta_bic, 1),
            "log_bf": np.round(log_bf, 1),
            "deviance_explained": marginal_de,
        }

    def _needs_clamp(kernel_obj):
        # Only for components already below the numerical pre-filter: the
        # full model's other parameters are already effectively optimized
        # as if this component didn't exist, so a refit would barely move
        # them -- skip it and evaluate the counterfactual directly.
        try:
            return float(kernel_obj.variance.numpy()) < VAR_CUTOFF_DEFAULT
        except (AttributeError, TypeError):
            return False

    def _clamp_result(kernel_obj):
        # Save/restore the underlying *unconstrained* variable directly,
        # not the constrained value via .assign(): round-tripping an
        # already-extreme constrained value back through the bijector's
        # inverse transform can overflow to +-Inf and fail Parameter.assign's
        # finiteness check, even though the value was valid before we
        # touched it.
        original_unconstrained = kernel_obj.variance.unconstrained_variable.numpy()
        try:
            kernel_obj.variance.assign(COMPONENT_CLAMP_VALUE)
            ll_clamped = model.log_posterior_density(data).numpy()
            mu_c, var_c = model.predict_y(data[0])
            null_lls_c, sub_mod_lls, _ = calc_deviance_explained(
                model=model,
                data=data,
                model_mu=mu_c,
                model_var=var_c,
                return_deviance_explained=False,
                aggregate=False,
                return_loglik=True,
            )
        finally:
            kernel_obj.variance.unconstrained_variable.assign(original_unconstrained)

        # The clamped component is fixed, not free: k_full - 1, matching
        # the same BIC formula the refit path uses, just without refitting.
        reduced_bic = calc_bic(
            loglik=ll_clamped, n=data[0].shape[0], k=k_full - 1
        )
        delta_bic = full_bic - reduced_bic
        log_bf = -0.5 * delta_bic

        denom = -2 * np.sum(null_lls_c - mod_lls)
        if denom != 0:
            marginal_de = (-2 * np.sum(sub_mod_lls - mod_lls)) / denom
            marginal_de = np.round(max(min(1, marginal_de), 0), 3)
        else:
            marginal_de = 0.0

        return {
            "delta_bic": np.round(delta_bic, 1),
            "log_bf": np.round(log_bf, 1),
            "deviance_explained": marginal_de,
        }

    def _refit_result_with_retry(make_reduced_kernel, max_attempts=2):
        # The refit itself can occasionally return a non-finite result --
        # observed to be a rare, non-reproducible numerical fault (not a
        # deterministic function of the data or of how many prior fits
        # happened in this process), so retrying with a fresh deepcopy can
        # succeed even though nothing about the inputs changed. If every
        # attempt is non-finite, return the last one anyway (with a
        # warning) rather than raise, so one bad component doesn't stop
        # every other component from being reported.
        result = None
        for _attempt in range(max_attempts):
            model_copy = gpflow.utilities.deepcopy(model)
            make_reduced_kernel(model_copy)
            result = _component_result(_refit(model_copy))
            if np.isfinite(result["log_bf"]):
                return result
        warnings.warn(
            "calc_feature_importance_components: refit produced a "
            f"non-finite log_bf after {max_attempts} attempts; returning "
            "it as-is. This component's result should be treated as "
            "unreliable."
        )
        return result

    detail_list = []
    if k.name == "sum":
        for k_idx in range(len(k.kernels)):
            target = k.kernels[k_idx]
            result = _clamp_result(target) if _needs_clamp(target) else None
            # Rare numerical edge case: the clamp evaluation itself can
            # produce a non-finite likelihood for some fitted states. Fall
            # back to the (more expensive but more robust) refit path
            # rather than propagate a NaN/Inf result.
            if result is None or not np.isfinite(result["log_bf"]):
                result = _refit_result_with_retry(
                    lambda m, idx=k_idx: m.kernel.kernels.pop(idx)
                )
            detail_list.append(result)

    else:
        # If there is just a single term, the reduced model is the
        # constant-kernel baseline (matches cut_kernel_components).
        if k.name == "constant":
            detail_list.append(
                {"delta_bic": 0.0, "log_bf": 0.0, "deviance_explained": 0.0}
            )
        else:
            result = _clamp_result(k) if _needs_clamp(k) else None
            if result is None or not np.isfinite(result["log_bf"]):
                result = _refit_result_with_retry(
                    lambda m: setattr(m, "kernel", gpflow.kernels.Constant())
                )
            detail_list.append(result)

    # Gather the final bit for leftover noise (always deviance-explained
    # units, matching the pre-refit contract).
    detail_list.append(
        {
            "delta_bic": None,
            "log_bf": None,
            "deviance_explained": np.round(1 - full_de, 3),
        }
    )

    if full_detail:
        return detail_list

    return feature_importance_detail_to_flat(detail_list, return_value)


def calc_empirical_pvalue(obs, null):
    """Empirical p-value(s) against a null pool.

    p = (1 + #{null >= obs}) / (1 + B) -- frozen decision (revision plan,
    item 4). Larger values of the statistic mean more evidence (matches the
    log_bf sign convention), so the tail counted is the upper tail.

    Parameters
    ----------
    obs : scalar or array-like
        Observed evidence statistic(s) (e.g. log_bf).
    null : array-like
        Pool of B null-distribution draws of the same statistic.

    Returns
    -------
    A scalar p-value if `obs` is scalar, else an np.ndarray aligned to
    `obs`.
    """
    obs_arr = np.atleast_1d(np.asarray(obs, dtype=float))
    null_arr = np.asarray(null, dtype=float)
    if null_arr.size == 0:
        raise ValueError("calc_empirical_pvalue: null pool is empty.")
    counts = (null_arr[None, :] >= obs_arr[:, None]).sum(axis=1)
    pvals = (1 + counts) / (1 + null_arr.size)
    return pvals.item() if np.ndim(obs) == 0 else pvals


def calc_bh_qvalues(pvalues):
    """Benjamini-Hochberg q-values (adjusted p-values).

    Standard step-up procedure: sort ascending, scale by n / rank, then
    take the cumulative minimum from the largest p-value down so q-values
    are monotone non-decreasing in the sorted p-value order.
    """
    pvals = np.asarray(pvalues, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    q_sorted = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    qvals = np.empty(n)
    qvals[order] = q_sorted
    return qvals


def empirical_null_bh(obs_values, null_values, obs_groups=None, null_groups=None):
    """Empirical-null p-values + BH q-values for a set of observed evidence
    statistics (e.g. per-(kernel, covariate) log_bf), against a pool of
    known-null draws of the same statistic.

    Frozen decision (revision plan, item 4): significance = empirical-null +
    BH, stratified per (kernel, covariate) pair. Pass `obs_groups`/
    `null_groups` (parallel arrays of group labels, e.g. "kernel:covariate"
    strings, one per entry in `obs_values`/`null_values`) to stratify: each
    group gets its own null pool and its own BH pass. Omitting them pools
    everything into one null distribution and one BH pass -- the plan's
    acceptance check uses this only as the contrast case, since pooling
    across strata that differ systematically over-rejects.

    Returns
    -------
    (pvalues, qvalues) : np.ndarray pair, aligned to `obs_values`.
    """
    obs_values = np.asarray(obs_values, dtype=float)
    null_values = np.asarray(null_values, dtype=float)

    if obs_groups is None:
        pvalues = calc_empirical_pvalue(obs_values, null_values)
        qvalues = calc_bh_qvalues(pvalues)
        return pvalues, qvalues

    obs_groups = np.asarray(obs_groups)
    null_groups = np.asarray(null_groups)
    pvalues = np.full(obs_values.shape, np.nan)
    qvalues = np.full(obs_values.shape, np.nan)

    for g in np.unique(obs_groups):
        obs_mask = obs_groups == g
        null_mask = null_groups == g
        if not np.any(null_mask):
            raise ValueError(
                f"empirical_null_bh: no null values found for group {g!r}."
            )
        p_g = calc_empirical_pvalue(obs_values[obs_mask], null_values[null_mask])
        pvalues[obs_mask] = p_g
        qvalues[obs_mask] = calc_bh_qvalues(p_g)

    return pvalues, qvalues


def calc_hardened_eb_qvalues(log_bf, groups=None, storey_lambda=0.5, null_offset=0.0):
    """Hardened empirical-Bayes (Efron two-groups) significance fallback.

    Used in place of `empirical_null_bh` when a formal null (known-null
    simulation components, or permutation replicates) is too expensive to
    obtain: estimates the null distribution of log_bf directly from its
    own negative side (a real signal should only push log_bf positive, so
    the negative side is assumed pure null), rather than requiring
    external null draws.

    Procedure:
    1. Fold every value below `null_offset` around `null_offset` and take
       the folded deviations' standard deviation (ddof=1).
    2. Correct for the fact that a folded half-normal's SD underestimates
       the full normal's SD, by the factor 1 / sqrt(1 - 2/pi) -- only
       valid if the fold point equals the true null mean, which is why
       `null_offset` matters for the scale estimate too, not just location.
    3. One-sided p-value under log_bf ~ Normal(null_offset, sigma_null).
    4. Storey's pi0 estimator (tuning parameter `storey_lambda`), capped
       at 1.
    5. pi0-weighted BH-style q-values (cumulative minimum from the largest
       p-value down).

    Parameters
    ----------
    log_bf : array-like
        One evidence statistic per observation (e.g. per metabolite, for a
        single (kernel, covariate) pair).
    groups : array-like, optional
        Parallel array of group labels (e.g. "kernel:covariate" strings).
        Pass to stratify: each group gets its own sigma_null, pi0, and
        q-value pass. Omit to run once over all of `log_bf`.
    storey_lambda : float
        Storey pi0 estimator tuning parameter. Default 0.5.
    null_offset : float or array-like, default 0.0
        Assumed mean of the null distribution of `log_bf`, one value per
        observation (broadcast if scalar). Defaults to 0 (the original,
        biased behavior). `log_bf = ΔLL - p` (p = parameters lost when a
        component is dropped: 2 for squared_exponential, 1 for
        lin/categorical) is not actually centered at 0 under the null --
        see docs/revision/FINDINGS.md, "T2 (continued)". Folding at the
        wrong point also invalidates the sqrt(1-2/pi) correction above
        (derived assuming the fold point is the true mean), so passing
        the right offset (e.g. `-p` per component) corrects sigma_null's
        scale, not just where p-values are centered. Pass per-component
        (not per-group) since kernel type -- and thus p -- can vary
        within a stratification group.

    Returns
    -------
    (pvalues, qvalues, diagnostics) : np.ndarray pair aligned to `log_bf`,
        plus a dict {group: {"sigma_null":..., "pi0_hat":...}} (group key
        is None if `groups` was not given).
    """
    log_bf = np.asarray(log_bf, dtype=float)
    null_offset = np.broadcast_to(
        np.asarray(null_offset, dtype=float), log_bf.shape
    )

    def _one_group(vals, offset):
        centered = vals - offset
        neg_vals = -centered[centered < 0]
        if neg_vals.size < 2:
            raise ValueError(
                "calc_hardened_eb_qvalues: fewer than 2 negative log_bf "
                "values available to estimate the null SD."
            )
        sigma_folded = neg_vals.std(ddof=1)
        if sigma_folded == 0:
            raise ValueError(
                "calc_hardened_eb_qvalues: negative log_bf values in this "
                "group are degenerate (zero spread, e.g. all tied after "
                "rounding) -- cannot estimate a null SD from them."
            )
        sigma_null = sigma_folded / np.sqrt(1 - 2 / np.pi)

        p = 1 - norm.cdf(centered / sigma_null)
        p = np.clip(p, 1e-12, 1.0)

        pi0_hat = min(np.mean(p > storey_lambda) / (1 - storey_lambda), 1.0)

        n = len(p)
        order = np.argsort(p)
        sorted_p = p[order]
        raw_q = pi0_hat * sorted_p * n / (np.arange(n) + 1)
        q_sorted = np.clip(np.minimum.accumulate(raw_q[::-1])[::-1], 0, 1)
        q = np.empty(n)
        q[order] = q_sorted

        return p, q, {"sigma_null": sigma_null, "pi0_hat": pi0_hat}

    if groups is None:
        p, q, diag = _one_group(log_bf, null_offset)
        return p, q, {None: diag}

    groups = np.asarray(groups)
    pvalues = np.full(log_bf.shape, np.nan)
    qvalues = np.full(log_bf.shape, np.nan)
    diagnostics = {}
    for g in np.unique(groups):
        mask = groups == g
        p_g, q_g, diag_g = _one_group(log_bf[mask], null_offset[mask])
        pvalues[mask] = p_g
        qvalues[mask] = q_g
        diagnostics[g] = diag_g

    return pvalues, qvalues, diagnostics


def individual_kernel_predictions(
    model,
    kernel_idx,
    data=None,
    product_term=False,
    X=None,
    white_noise_amt=1e-6,
    predict_type="func",
    marginal=True,
    num_samples=100,
    model_data=None,
    latent=False,
):
    """Predict contribution from individual kernel component.

    Parameters
    ----------
    model : gpflow.model

    kernel_idx : Integer

    X : Numpy array for prediction points

    white_noise_amt : Float
        Amount of diagonal noise to add to covariance matricies

    predict_type : String
        Add Gaussian noise from likelihood function?
        Options: ["func", "latent", "mean"]

    num_samples : Integer
        Number of samples to draw from the posterior component

    Attributes
    ----------

    """

    # Set model data if not supplied
    if hasattr(model, "data") is False and data is None:
        if latent is True:
            model_data = (
                model.inducing_variable.inducing_variable_list[
                    kernel_idx
                ].Z.numpy(),
                model.q_mu.numpy()[:, kernel_idx].reshape(-1, 1),
            )
        else:
            if model.inducing_variable is None:
                assert (
                    model_data is not None
                ), "Need to supply model_data argument for this model type."
    elif data is not None:
        model_data = data
    else:
        model_data = model.data

    # Overwrite model data with inducing variables if they exist
    if model.inducing_variable is not None:
        model_data = (model.inducing_variable.Z, model.q_mu)

    # Copy model component of interest
    sub_model = gpflow.utilities.deepcopy(model)

    # Make sure we have additive components, otherwise return the full model
    if sub_model.kernel.name == "sum":

        # Check to make sure there is an appropriate kernel index
        if kernel_idx >= len(sub_model.kernel.kernels):
            raise ValueError(
                "Not enough kernel components for index requested!"
            )

        # TODO: Show that using the independent kernel is bad!
        # Now subset the copied model to the specific kernel component
        sub_model.kernel = sub_model.kernel.kernels[kernel_idx]

    # # Then generate predictions
    # pred_mu, pred_var = sub_model.predict_f(X)
    # _, pred_cov = sub_model.predict_f(X, full_cov=True)
    # sample_fns = tf.transpose(
    #     sub_model.predict_f_samples(X, num_samples=num_samples)[:, :, 0]
    # )

    # # Transform output as needed
    # if predict_type == "mean":
    #     sample_fns = model.likelihood._conditional_mean(X=X, F=sample_fns)
    #     pred_var = model.likelihood._conditional_variance(X=X, F=pred_mu)
    #     pred_mu = model.likelihood._conditional_mean(X=X, F=pred_mu)
    #     pred_cov = None

    # return pred_mu, pred_var, sample_fns, pred_cov

    # TODO: Double check to see if we need this product block for anything
    # # Only pull of kernel of interest if there are multiple kernels
    # # Also need to deal with product term
    # if hasattr(sub_model.kernel, "kernels"):
    #     if product_term:
    #         sub_model.kernel = gpflow.kernels.Product([
    #             sub_model.kernel.kernels[kernel_idx],
    #             sub_model.kernel.kernels[kernel_idx+1]
    #         ])
    #     else:
    #         print(f"{sub_model.kernel.kernels=}")
    #         sub_model.kernel = sub_model.kernel.kernels[kernel_idx]
    # pred_x = model_data[0] if X is None else X

    # If there is only one kernel component then return
    # standard marginal prediction
    if sub_model.kernel.name != "sum":
        pred_mu, pred_var = sub_model.predict_f(X)
        _, pred_cov = sub_model.predict_f(X, full_cov=True)
        sample_fns = tf.transpose(
            sub_model.predict_f_samples(X, num_samples=num_samples)[:, :, 0]
        )
    else:

        if marginal is True:
            pred_mu, pred_var = sub_model.predict_f(X)
            _, pred_cov = sub_model.predict_f(X, full_cov=True)
            sample_fns = tf.transpose(
                sub_model.predict_f_samples(X, num_samples=num_samples)[
                    :, :, 0
                ]
            )
        else:
            # Build each part of the covariance matrix
            if latent is True:
                sigma_21 = tf.cast(
                    model.kernel.latent_kernels[kernel_idx].K(
                        X=model_data[0], X2=X
                    ),
                    gpflow.default_float(),
                )
                sigma_11 = tf.cast(
                    model.kernel.latent_kernels[kernel_idx].K(X=X), gpflow.default_float()
                )
            elif model.kernel.name == "sum":
                sigma_21 = tf.cast(
                    model.kernel.kernels[kernel_idx].K(X=model_data[0], X2=X),
                    gpflow.default_float(),
                )
                sigma_11 = tf.cast(
                    model.kernel.kernels[kernel_idx].K(X=X), gpflow.default_float()
                )
            else:
                sigma_21 = tf.cast(
                    model.kernel.K(X=model_data[0], X2=X), gpflow.default_float()
                )
                sigma_11 = tf.cast(model.kernel.K(X=X), gpflow.default_float())

            if latent is True:
                sigma_22 = tf.cast(
                    model.kernel.latent_kernels[kernel_idx](X=model_data[0]),
                    gpflow.default_float(),
                )
            else:
                sigma_22 = tf.cast(model.kernel.K(X=model_data[0]), gpflow.default_float())
            sigma_12 = tf.transpose(sigma_21)

            # Figure out white noise amount to add to diag if none given
            if white_noise_amt is None:
                # Get min eigenvalue to make sure we can invert the matrix
                min_ev = np.min(np.linalg.eigvalsh(sigma_22))
                if min_ev < 0:
                    white_noise_amt = abs(min_ev)
                else:
                    white_noise_amt = 0
                # white_noise_amt = np.tril(sigma_11, k=-1).max()
            sigma_22 += tf.linalg.diag(
                tf.repeat(f64(white_noise_amt), model_data[0].shape[0])
            )

            # Invert sigma_22
            # Try LU decomposition first
            try:
                inv_sigma_22 = tfp.math.lu_matrix_inverse(
                    *tf.linalg.lu(sigma_22)
                )
            except ValueError:
                print("Warning - Approximating the covariance inverse")
                inv_sigma_22 = tf.linalg.pinv(sigma_22)

            # Now calculate mean and variance
            if latent is True:
                pred_mu = np.zeros((X.shape[0], 1)) + tf.matmul(
                    a=tf.matmul(
                        a=sigma_12,
                        b=inv_sigma_22,  # b=tf.linalg.inv(sigma_22)),
                    ),
                    b=(
                        model.q_mu.numpy()[:, kernel_idx].reshape(-1, 1)
                        - np.zeros((model_data[0].shape[0], 1))
                    ),
                )
            else:
                if model.mean_function.name == "zero":
                    mu1 = np.zeros(shape=(X.shape[0], 1))
                    mu2 = np.zeros(shape=(model_data[0].shape[0], 1))
                elif model.mean_function.name == "constant":
                    mu1 = np.repeat(
                        model.mean_function.c.numpy(), X.shape[0]
                    ).reshape(-1, 1)
                    mu2 = np.repeat(
                        model.mean_function.c.numpy(), model_data[0].shape[0]
                    ).reshape(-1, 1)
                else:
                    raise NotImplementedError(
                        "Cannot handle mean_function beyond (none, constant)"
                    )

                # Calculate posterior mean
                pred_mu = mu1 + tf.matmul(
                    a=tf.matmul(a=sigma_12, b=inv_sigma_22),
                    b=(model_data[1] - mu2),
                )

            # Covariance function
            pred_cov = sigma_11 - tf.matmul(
                a=sigma_12,
                b=tf.matmul(
                    a=inv_sigma_22,
                    # a=tf.linalg.inv(sigma_22),
                    b=sigma_21,
                ),
            )

            # Add uncertainty from inducing variables if present
            if model.inducing_variable is not None:
                pred_cov += tf.matmul(
                    a=sigma_12,
                    b=tf.matmul(
                        a=inv_sigma_22,
                        b=tf.matmul(
                            a=model.q_sqrt,
                            b=tf.matmul(a=inv_sigma_22, b=sigma_21),
                        ),
                    ),
                )

            # Variance component
            pred_var = tf.linalg.diag_part(pred_cov)

            # Also pull some function samples (if covariance is stable!)
            try:
                posterior_dist = tfp.distributions.MultivariateNormalTriL(
                    loc=tf.transpose(pred_mu),
                    scale_tril=tf.linalg.cholesky(pred_cov),
                    validate_args=True,
                )
                sample_fns = posterior_dist.sample(sample_shape=num_samples)
                sample_fns = tf.transpose(
                    tf.reshape(sample_fns, (num_samples, -1))
                )
            except tf.errors.InvalidArgumentError:
                sample_fns = tf.repeat(pred_mu, num_samples, axis=1)

    # Transform output as needed
    if predict_type == "mean":
        sample_fns = model.likelihood._conditional_mean(X=X, F=sample_fns)
        pred_var = model.likelihood._conditional_variance(X=X, F=pred_mu)
        pred_mu = model.likelihood._conditional_mean(X=X, F=pred_mu)
        pred_cov = None

    return pred_mu, pred_var, sample_fns, pred_cov


def freeze_variance_parameters(kernel):
    if hasattr(kernel, "variance"):
        gpflow.utilities.set_trainable(kernel.variance, False)
        return None
    elif kernel.name in ["sum", "product", "linear_coregionalization"]:
        for k in kernel.kernels:
            # print(f"working on kernel {k}")
            freeze_variance_parameters(k)
    elif kernel.name == "periodic":
        freeze_variance_parameters(kernel.base_kernel)


def gp_likelihood_crosswalk(likelihood_str):
    if likelihood_str == "gaussian":
        return gpflow.likelihoods.Gaussian()
    elif likelihood_str == "poisson":
        return gpflow.likelihoods.Poisson()
    elif likelihood_str in ["binomial", "bernoulli"]:
        return gpflow.likelihoods.Bernoulli()
    elif likelihood_str == "gamma":
        return gpflow.likelihoods.Gamma()
    elif likelihood_str == "negativebinomial":
        return NegativeBinomial()
    elif likelihood_str == "zeroinflated_negativebinomial":
        return ZeroInflatedNegativeBinomial()
    else:
        print(
            "Not sure what likelihood requested. Can use 'gaussian',"
            " 'poisson', 'binomial', 'bernoulli', 'gamma',"
            " 'negativebinomial', "
            " and 'zeroinflated_negativebinomial'."
        )
        return None


def find_variance_components(
    kern, sum_reduce=True, penalize_factor_prod=1, return_numpy=True
):
    """Retrieve the variance parameter of all kernel components recursively."""
    # print(kern.name)
    if kern.name == "sum":
        var_list = np.stack(
            [
                find_variance_components(kern=x, sum_reduce=sum_reduce)
                for x in kern.kernels
            ]
        )
        if sum_reduce:
            return np.sum(var_list)
        else:
            return var_list
    elif kern.name == "product":
        return np.array(
            [
                penalize_factor_prod
                * np.prod(
                    [
                        find_variance_components(x, sum_reduce)
                        for x in kern.kernels
                    ]
                )
            ]
        )
    elif kern.name == "linear_coregionalization":
        if return_numpy:
            temp_weights = kern.W.numpy()
        else:
            temp_weights = kern.W

        if sum_reduce:
            return np.sum(np.abs(temp_weights))
        else:
            return np.abs(temp_weights)
    else:
        if kern.name == "periodic":
            if return_numpy:
                return np.array([kern.base_kernel.variance.numpy()])
            else:
                return np.array([kern.base_kernel.variance])
        elif kern.name == "empty":
            return np.zeros(1)
        else:
            if return_numpy:
                return np.array([kern.variance.numpy()])
            else:
                return np.array([kern.variance])


def find_variance_components_tf(
    kern,
    sum_reduce=True,
    penalize_factor_prod=1,
):
    """Retrieve the variance parameter of all kernel components recursively."""
    # print(kern.name)
    if kern.name == "sum":
        var_list = tf.stack(
            [
                find_variance_components_tf(kern=x, sum_reduce=sum_reduce)
                for x in kern.kernels
            ]
        )
        if sum_reduce:
            return tf.reduce_sum(var_list)
        else:
            return var_list
    elif kern.name == "product":
        return penalize_factor_prod * tf.reduce_prod(
            tf.stack(
                [
                    find_variance_components_tf(x, sum_reduce)
                    for x in kern.kernels
                ]
            )
        )
    elif kern.name == "linear_coregionalization":
        temp_weights = kern.W

        if sum_reduce:
            return tf.reduce_sum(tf.abs(temp_weights))
        else:
            return tf.abs(temp_weights)
    else:
        if kern.name == "periodic":
            return kern.base_kernel.variance
        elif kern.name == "empty":
            return tf.zeros(shape=(), dtype=gpflow.default_float())
        else:
            return tf.reduce_sum(kern.variance)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar
    given as argument.

    Source: (
        https://stackoverflow.com/questions/24983493
        /tracking-progress-of-joblib-parallel-execution
    )
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def keep_kernel_lengthscale_(kernel_component, X):
    """Check to see if we should drop kernel component based on lengthscale."""
    # Drill down to base kernel if periodic
    if kernel_component.name == "periodic":
        kernel_component = kernel_component.base_kernel

    # Test to make sure this kernel component is governed by lengthscale
    if hasattr(kernel_component, "lengthscales") is False:
        return True

    # Pull off active dimension
    active_index = kernel_component.active_dims[0]

    # Check to see range of relevant input dimension
    var_range = 3 * np.ptp(X[:, active_index])

    # See if lengthscale is larger than range of input
    return kernel_component.lengthscales.numpy() < var_range


def search_through_kernel_list_(kernel_list, list_type="sum", X=None):
    """Only keep kernel components that meet criteria.
    Account for sum versus product kernels.
    """
    out_list = []
    for i, k in enumerate(kernel_list):
        if k.name == "product":
            prod_out = search_through_kernel_list_(
                k.kernels, list_type="product", X=X
            )
            out_list.append(prod_out)
        else:
            keep_component = keep_kernel_lengthscale_(k, X)
            if keep_component:
                out_list.append(k)

    # Stich together components based on type
    out_kernel = None
    if len(out_list) > 1:
        if list_type == "sum":
            out_kernel = gpflow.kernels.Sum(out_list)
        elif list_type == "product":
            out_kernel = gpflow.kernels.Product(out_list)
    elif len(out_list) == 1:
        out_kernel = out_list[0]
    else:
        out_kernel = Empty()

    return out_kernel


class ParallelTqdm(Parallel):
    """joblib.Parallel, but with a tqdm progressbar

    Source: https://github.com/joblib/joblib/issues/972

    Additional parameters:
    ----------------------
    total_tasks: int, default: None
        the number of expected jobs. Used in the tqdm progressbar.
        If None, try to infer from the length of the called iterator, and
        fallback to use the number of remaining items as soon as we finish
        dispatching.
        Note: use a list instead of an iterator if you want the total_tasks
        to be inferred from its length.

    desc: str, default: None
        the description used in the tqdm progressbar.

    disable_progressbar: bool, default: False
        If True, a tqdm progressbar is not used.

    show_joblib_header: bool, default: False
        If True, show joblib header before the progressbar.

    Removed parameters:
    -------------------
    verbose: will be ignored


    Usage:
    ------
    >>> from joblib import delayed
    >>> from time import sleep
    >>> ParallelTqdm(n_jobs=-1)([delayed(sleep)(.1) for _ in range(10)])
    80%|████████  | 8/10 [00:02<00:00,  3.12tasks/s]

    """

    def __init__(
        self,
        *,
        total_tasks: int | None = None,
        desc: str | None = None,
        disable_progressbar: bool = False,
        show_joblib_header: bool = False,
        **kwargs,
    ):
        if "verbose" in kwargs:
            raise ValueError(
                "verbose is not supported. "
                "Use show_progressbar and show_joblib_header instead."
            )
        super().__init__(verbose=(1 if show_joblib_header else 0), **kwargs)
        self.total_tasks = total_tasks
        self.desc = desc
        self.disable_progressbar = disable_progressbar
        self.progress_bar: tqdm.tqdm | None = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                # try to infer total_tasks from the length of the called iterator
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            # call parent function
            return super().__call__(iterable)
        finally:
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator):
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = tqdm.tqdm(
                desc=self.desc,
                total=self.total_tasks,
                disable=self.disable_progressbar,
                unit="tasks",
            )
        # call parent function
        return super().dispatch_one_batch(iterator)

    dispatch_one_batch.__doc__ = Parallel.dispatch_one_batch.__doc__

    def print_progress(self):
        """Display the process of the parallel execution using tqdm"""
        # if we finish dispatching, find total_tasks from the number of remaining items
        if self.total_tasks is None and self._original_iterator is None:
            self.total_tasks = self.n_dispatched_tasks
            self.progress_bar.total = self.total_tasks
            self.progress_bar.refresh()
        # update progressbar
        self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)


def run_ray_process(
    num_jobs=-1,
    num_entities_per_round=5,
    model_output_names=[],
    func=None,
    stored_func_args={},
    include_ray_dashboard=False,
):

    # Set up output object
    objs = {}

    # Set up number of processes and partition work out
    if num_jobs == -1:
        num_processes = None
        num_feats_per_round = num_entities_per_round * psutil.cpu_count()
    else:
        num_processes = num_jobs
        num_feats_per_round = num_entities_per_round * num_processes

    grouped_feat_list = [
        model_output_names[x : x + num_feats_per_round]
        for x in range(0, len(model_output_names), num_feats_per_round)
    ]

    # Set up tracker
    remote_tqdm = ray.remote(tqdm_ray.tqdm)

    # Start running through partitions of tasks
    num_feats = len(model_output_names)
    print(f"Building {num_feats} models...")
    start_time = time.time()
    c = 0

    for i in grouped_feat_list:
        # Initialize ray
        try:
            ray.init(
                num_cpus=num_processes,
                include_dashboard=include_ray_dashboard,
                configure_logging=False,
            )
        except RuntimeError:
            ray.shutdown()
            ray.init(
                num_cpus=num_processes,
                include_dashboard=include_ray_dashboard,
                configure_logging=False,
            )

        # Store data in shared data store
        for k, v in stored_func_args.items():
            k = ray.put(v)

        # self_X = ray.put(self.X)
        # self_Y = ray.put(self.Y)
        # self_likelihood = ray.put(self.likelihood)
        # if hasattr(self, "Y_stds"):
        #     self_Y_stds = ray.put(self.Y_stds)
        # else:
        #     self_Y_stds = ray.put(None)

        # Load function
        func_remote = ray.remote(func)

        # Create progress bar
        bar = remote_tqdm.remote(total=len(i))

        # Retrieve output from processes
        out = ray.get(
            [
                func_remote.remote(
                    # self_X,
                    # self_Y,
                    # self_likelihood,
                    # self_Y_stds,
                    **stored_func_args,
                    feat=feat,
                    bar=bar,
                )
                for feat in i
            ]
        )

        # Save output to returned object
        for feat, mod in zip(i, out):
            objs[feat] = mod

        # Clean up ray
        bar.close.remote()
        ray.shutdown()

        # Add number of finished models
        c += len(i)

        # Print output
        prop_done = int(np.round(100 * c / num_feats))
        elapsed_time = np.round((time.time() - start_time) / 60, 1)
        print(
            f"Finished {c} models ({prop_done}%),",
            f"elapsed time: {elapsed_time} minutes",
        )

    return objs


