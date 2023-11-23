import gpflow
import numpy as np
from gpflow.inducing_variables import SeparateIndependentInducingVariables
from tensorflow_probability import distributions as tfd

from .model_types_DEPR import SVPGPR
from .utilities import (
    calc_bic,
    f64,
    freeze_variance_parameters,
    gp_likelihood_crosswalk,
    print_kernel_names,
)


def kernel_test_reg(
    X,
    Y,
    k,
    num_restarts=5,
    random_init=True,
    verbose=False,
    likelihood="gaussian",
    lasso=False,
    lam=0,
    gam=0,
    base_variances=None,
    max_iter=50000,
    use_priors=True,
    keep_data=False,
    X_holdout=None,
    Y_holdout=None,
    split=False,
    freeze_inducing=False,
    freeze_variances=False,
    random_seed=None,
    num_inducing_points=500,
):
    """
    This function evaluates a particular kernel selection on a set of data.

    Inputs:
        X (array): design matrix
        Y (array): output matrix
        k (gpflow kernel): specified kernel

    Outputs:
        m (gpflow model): fitted GPflow model

    """

    k = gpflow.utilities.deepcopy(k)

    # Randomize initial values for a number of restarts
    if random_seed is not None:
        np.random.seed(random_seed)
    best_loglik = -np.Inf
    best_model = None
    # better_model_seen = False

    for i in range(num_restarts):
        # # Check to see if it worthwhile to keep restarting
        # if better_model_seen is False and i > ceil((i+1)/2):
        #     if verbose:
        #         print(f'Tried {i+1} restarts and nothing better seen.')
        #         print('Now stopping!')
        #     break

        # Specify model
        if lasso:
            # Go from string to gpflow likelihood object
            if isinstance(likelihood, str):
                gp_likelihood = gp_likelihood_crosswalk(likelihood)
            else:
                gp_likelihood = likelihood

            # m = PGPR(
            #     data=(X, Y),
            #     kernel=k,
            #     lam=lam,
            #     gam=gam,
            #     base_variances=base_variances
            # )

            if num_inducing_points is None:
                num_inducing_points = 500
            elif num_inducing_points > X.shape[0]:
                num_inducing_points = X.shape[0]

            # Break out the number of latent GPs
            # print(k)
            if hasattr(k, "W"):
                # print(f"W shape = {k.W.shape}")
                # print(f"Y shape = {Y.shape}")
                num_latent = k.W.shape[1]
            else:
                num_latent = 1

            if verbose:
                print(f"Using likelihood: {gp_likelihood}")

            # Use inducing points if there are latent processes
            if num_latent > 1:
                m = SVPGPR(
                    X=X,
                    Y=Y,
                    kernel=k,
                    lam=0,  # lam,
                    gam=gam,
                    base_variances=base_variances,
                    likelihood=gp_likelihood,
                    inducing_variable=SeparateIndependentInducingVariables(
                        [
                            gpflow.inducing_variables.InducingPoints(
                                # X_[:num_inducing_points,:]
                                X_[
                                    np.random.randint(
                                        low=0,
                                        high=X.shape[0],
                                        size=num_inducing_points,
                                    ),
                                    :,
                                ]
                            )
                            for X_ in [X.copy() for _ in range(num_latent)]
                        ]
                    ),
                    # inducing_variable=gpflow.inducing_variables.SharedIndependentInducingVariables(
                    #     gpflow.inducing_variables.InducingPoints(
                    #         X[np.random.randint(
                    #               low=0,
                    #               high=X.shape[0],
                    #               size=num_inducing_points
                    #           ), :].copy()
                    #     )
                    # ),
                    num_latent_gps=num_latent,
                )
            else:
                m = SVPGPR(
                    X=X,
                    Y=Y,
                    kernel=k,
                    lam=0,  # lam,
                    gam=gam,
                    base_variances=base_variances,
                    likelihood=gp_likelihood,
                    num_latent_gps=num_latent,
                )
        elif likelihood == "gaussian":
            m = gpflow.models.GPR(
                #             m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k,
            )  # +gpflow.kernels.Constant())#,
            # mean_function=gpflow.mean_functions.Constant())#,
        #                 likelihood=gpflow.likelihoods.Gaussian())
        elif likelihood == "exponential":
            m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k,
                # mean_function=gpflow.mean_functions.Constant(),
                likelihood=gpflow.likelihoods.Exponential(),
            )
        elif likelihood == "poisson":
            m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k,
                # mean_function=gpflow.mean_functions.Constant(),
                likelihood=gpflow.likelihoods.Poisson(),
            )
        elif likelihood == "gamma":
            m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k,
                # mean_function=gpflow.mean_functions.Constant(),
                likelihood=gpflow.likelihoods.Gamma(),
            )
        elif likelihood == "bernoulli":
            m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k,  # +gpflow.kernels.Constant(),
                # mean_function=gpflow.mean_functions.Constant(),
                likelihood=gpflow.likelihoods.Bernoulli(),
            )
        else:
            print("Unknown likelihood requested.")

        # Freeze requested model parameters from training
        if freeze_inducing is True:
            gpflow.utilities.set_trainable(
                m.inducing_variable.inducing_variables, False
            )

        if freeze_variances is True:
            freeze_variance_parameters(m.kernel)

        if lam > 0:
            if num_latent > 1:
                print(f"Setting laplace(0, {1/lam}) prior on W!")
                m.kernel.W.prior = tfd.Laplace(
                    loc=f64(0), scale=f64(1.0 / lam)
                )
            else:
                print(f"Setting laplace(0, {1/lam}) prior on variances!")
                for param_name, param_val in gpflow.utilities.parameter_dict(
                    m
                ).items():
                    if "kernel" in param_name and "variance" in param_name:
                        param_val.prior = tfd.Laplace(
                            loc=f64(0), scale=f64(1.0 / lam)
                        )
            # m.kernel.W.prior = tfd.Laplace(
            #     loc=f64(0),
            #     scale=tfd.Gamma(concentration=f64(1), rate=f64(1))
            # )

        # Set current model to best
        if best_model is None:
            best_model = m

        # Uncomment to view base model being optimized
        # print(gpflow.utilities.print_summary(m))

        # Set priors
        # We don't want to use priors if we are regularizing
        # if use_priors is True: # and lasso is False:
        #     for p in m.parameters:
        #         # We don't want to mess with the Q points in VGP model
        #         if len(p.shape) <= 1:
        #             if p.name == "identity":
        #                 p.prior = tfd.Uniform(f64(-100), f64(100))
        #             else:
        #                 p.prior = tfd.Uniform(f64(0), f64(100))

        if use_priors is True:
            for param_name, param_val in gpflow.utilities.parameter_dict(
                m
            ).items():
                if "kernel" in param_name:
                    if "variance" not in param_name and "W" not in param_name:
                        param_val.prior = tfd.Uniform(f64(0), f64(10))

        # Randomize initial values if not trained already
        if random_init:
            for p in m.kernel.trainable_parameters:
                # Should we actually not have this requirement?
                if len(p.shape) <= 1 or p.shape == k.W.shape:
                    unconstrain_vals = np.random.normal(
                        size=p.numpy().size
                    ).reshape(p.numpy().shape)
                    p.assign(p.transform_fn(unconstrain_vals))

            for p in m.likelihood.trainable_parameters:
                if len(p.shape) <= 1:
                    unconstrain_vals = np.random.normal(
                        size=p.numpy().size
                    ).reshape(p.numpy().shape)
                    p.assign(p.transform_fn(unconstrain_vals))

            # m.q_mu = np.random.normal(size=m.q_mu.shape)

        # Set prior on W
        # m.kernel.W.prior = tfp.distributions.Exponential(f64(1))

        # Optimization step for hyperparameters
        try:
            if m.name == "svpgpr":
                opt_res = gpflow.optimizers.Scipy().minimize(
                    m.training_loss_closure((X, Y)),
                    variables=m.trainable_variables,
                    # method="Nelder-Mead",
                    options={"maxiter": max_iter, "maxcor": 50, "ftol": 1e-10},
                )
            else:
                opt_res = gpflow.optimizers.Scipy().minimize(
                    m.training_loss,
                    m.trainable_variables,
                    # method="Nelder-Mead",
                    options={"maxiter": max_iter},
                )
            # adam_opt_params(m)
            # scipy_opt_params(m)

            # Check if model converged
            if opt_res["success"] is False:
                if verbose:
                    print("Warning: optimizer did not converge!")

        except Exception as e:
            if verbose:
                print(f"Optimization not successful, skipping. Error: {e}")
                print(i)
            if best_model is None and i == num_restarts - 1:
                return m, -1 * best_loglik
            continue

        #         print(opt_results)
        # Now check to see if this is invertible
        try:
            m_, v_ = m.predict_y(m.data[0])
        except Exception:
            if verbose:
                print("Covariance matrix not invertible, removing model.")
            # If not invertible then revert back to best model, unless last try
            if best_model is None and i == num_restarts - 1:
                return best_model, -1 * best_loglik
            else:
                m = best_model

        # Check if better values found and save if so
        #         if m.log_marginal_likelihood() > best_loglik:
        if m.name == "svpgpr":
            cur_loglik = m.log_posterior_density(data=(X, Y)).numpy()
        else:
            cur_loglik = m.log_posterior_density().numpy()
        if cur_loglik > best_loglik:
            best_loglik = cur_loglik
            best_model = gpflow.utilities.deepcopy(m)
            # print(opt_res)
            # better_model_seen = True
            if verbose:
                print(f"New best log likelihood: {best_loglik}")
        else:
            del m

    #     # Set hyperparameters to best found values
    #     for l in range(len(m.trainable_parameters)):
    #         print(best_params[l])
    #         m.trainable_parameters[l].assign(best_params[l])

    # Return none and worst BIC if we can't fit a single
    if best_model is None:
        return best_model, -1 * best_loglik

    # Calculate information criteria
    if split:
        yhat_holdout = best_model.predict_f(X_holdout)
        estimated_loglik = (
            best_model.likelihood.predict_log_density(
                yhat_holdout[0], yhat_holdout[1], Y_holdout
            )
            .numpy()
            .sum()
        )
        bic = round(-1 * estimated_loglik, 2)
    else:
        estimated_loglik = (
            best_loglik  # best_model.log_posterior_density().numpy()
        )

        bic = round(
            calc_bic(
                #         loglik=best_model.log_marginal_likelihood().numpy(),
                loglik=estimated_loglik,
                n=X.shape[0],
                k=len(best_model.trainable_parameters),
            ),
            2,
        )

    # Print out info if requested
    if verbose:
        print(f"Model: {print_kernel_names(k)}, BIC: {bic}")

    # Delete data from model object
    if not keep_data:
        best_model.data = None

    # Return fitted GP model and bic
    # Predictions
    #     print(best_model.predict_f(X))
    return best_model, bic
