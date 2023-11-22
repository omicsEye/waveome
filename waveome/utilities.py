import contextlib

import gpflow
import joblib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import set_trainable
from tensorflow_probability import distributions as tfd

from .likelihoods import ZeroInflatedNegativeBinomial

# from multiprocessing import Value


f64 = gpflow.utilities.to_default_float


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
    # return k*np.log(n)-2*loglik
    return 2 * k - 2 * loglik


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


def calc_rsquare(m):
    """
    Calculate the r-squared values of each kernel component.
    """

    # Save output list
    rsq = []

    # Pull off data from stored model
    X = m.data[0].numpy()
    Y = m.data[1].numpy()

    # Make copy of model
    m_copy = gpflow.utilities.deepcopy(m)

    # Calculate the mean of the outcome
    Y_bar = Y.mean()

    # Calculate sum of squares error
    sse = np.sum((Y - Y_bar) ** 2)

    # Calculate overall model predictions
    mu_all_hat, var_all_hat = m.predict_y(X)
    ssr_total = np.sum((Y - mu_all_hat) ** 2)
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
                model=m, kernel_idx=k_idx, X=m.data[0]
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


def calc_residuals(m, X=None, Y=None):
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

    # Calculate standardized residuals
    resids = ((tf.cast(Y, tf.float64) - mean_resp) / np.sqrt(var_resp)).numpy()

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


def individual_kernel_predictions(
    model,
    kernel_idx,
    X=None,
    white_noise_amt=1e-6,
    predict_type="func",
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

    num_samples : Integer
                  Number of samples to draw from the posterior component

    Attributes
    ----------

    """

    # Set model data if not supplied
    if hasattr(model, "data") is False:
        if latent is True:
            model_data = (
                model.inducing_variable.inducing_variable_list[
                    kernel_idx
                ].Z.numpy(),
                model.q_mu.numpy()[:, kernel_idx].reshape(-1, 1),
            )
        else:
            assert (
                model_data is not None
            ), "Need to supply model_data argument for this model type."
    else:
        model_data = model.data

    # Copy model component of interest
    sub_model = gpflow.utilities.deepcopy(model)

    # Only pull of kernel of interest if there are multiple kernels
    if hasattr(sub_model.kernel, "kernels"):
        sub_model.kernel = sub_model.kernel.kernels[kernel_idx]

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
        # Build each part of the covariance matrix
        if latent is True:
            sigma_21 = tf.cast(
                model.kernel.latent_kernels[kernel_idx].K(
                    X=model_data[0], X2=X
                ),
                tf.float64,
            )
            sigma_11 = tf.cast(
                model.kernel.latent_kernels[kernel_idx].K(X=X), tf.float64
            )
        elif model.kernel.name == "sum":
            sigma_21 = tf.cast(
                model.kernel.kernels[kernel_idx].K(X=model.data[0], X2=X),
                tf.float64,
            )
            sigma_11 = tf.cast(
                model.kernel.kernels[kernel_idx].K(X=X), tf.float64
            )
        else:
            sigma_21 = tf.cast(
                model.kernel.K(X=model.data[0], X2=X), tf.float64
            )
            sigma_11 = tf.cast(model.kernel.K(X=X), tf.float64)

        if latent is True:
            sigma_22 = tf.cast(
                model.kernel.latent_kernels[kernel_idx](X=model_data[0]),
                tf.float64,
            )
        else:
            sigma_22 = tf.cast(model.kernel.K(X=model_data[0]), tf.float64)
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
            tf.repeat(f64(white_noise_amt), model.data[0].shape[0])
        )

        # Invert sigma_22
        # Try LU decomposition first
        try:
            inv_sigma_22 = tfp.math.lu_matrix_inverse(*tf.linalg.lu(sigma_22))
        except ValueError:
            print("Warning - Approximating the covariance inverse")
            inv_sigma_22 = tf.linalg.pinv(sigma_22)

        # Now calculate mean and variance
        if latent is True:
            pred_mu = np.zeros((X.shape[0], 1)) + tf.matmul(
                a=tf.matmul(
                    a=sigma_12, b=inv_sigma_22  # b=tf.linalg.inv(sigma_22)),
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
        # Variance component
        pred_var = tf.linalg.diag_part(pred_cov)

        # Also pull some function samples
        # posterior_dist = tfp.distributions.MultivariateNormalFullCovariance(
        #     loc=tf.transpose(pred_mu),
        #     covariance_matrix=pred_cov,
        #     )
        # Need to update this to silence tensorflow warning
        posterior_dist = tfp.distributions.MultivariateNormalTriL(
            loc=tf.transpose(pred_mu), scale_tril=tf.linalg.cholesky(pred_cov)
        )
        sample_fns = posterior_dist.sample(sample_shape=num_samples)
        sample_fns = tf.transpose(tf.reshape(sample_fns, (num_samples, -1)))

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
    elif likelihood_str == "zeroinflated_negativebinomial":
        return ZeroInflatedNegativeBinomial()
    else:
        print(
            "Not sure what likelihood requested. Can use 'gaussian',"
            " 'poisson', 'binomial', and 'gamma'."
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
        return (
            penalize_factor_prod *
            tf.reduce_prod(
                    tf.stack(
                        [
                            find_variance_components_tf(x, sum_reduce)
                            for x in kern.kernels
                        ]
                    )
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
            return tf.zeros(1, shape=())
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
