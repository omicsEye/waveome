import gpflow
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import numpy as np
from math import ceil
import os
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from gpflow.utilities import add_likelihood_noise_cov, set_trainable
from gpflow.logdensities import multivariate_normal
from gpflow.base import RegressionData
from .kernels import Categorical
from .utilities import (
    calc_bic,
    print_kernel_names,
    freeze_variance_parameters,
    gp_likelihood_crosswalk
)

f64 = gpflow.utilities.to_default_float
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class PGPR(gpflow.models.GPR):
    def __init__(self, data, kernel, mean_function = None, 
                 noise_variance = 1.0, lam = 1.0, 
                 base_variances = None, gam = 1.0):
        super().__init__(data, kernel, mean_function, noise_variance)
        self.lam = lam
        self.gam = gam        
        
        # Set base variances to ones if none given
        if base_variances is None:
            num_vars = len(find_variance_components(kernel, sum_reduce=False))
            self.base_variances = np.ones(shape=num_vars)
        else:
            self.base_variances = base_variances

    def set_lambda(self, new_lam):
        self.lam = new_lam
        
    def set_gamma(self, new_gam):
        self.gam = new_gam
        
    def log_marginal_likelihood(self, penalize=True) -> tf.Tensor:
        """
        Computes the log marginal likelihood.
        .. math::
            \log p(Y | \theta).
        """
        
        X, Y = self.data
        K = self.kernel(X)
        ks = add_likelihood_noise_cov(K, self.likelihood, X)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        pen_log_prob = (tf.reduce_mean(log_prob) -
            tf.reduce_sum((len(X)) * self.lam *
                          (1/(self.base_variances)**self.gam) * 
                          find_variance_components(self.kernel, sum_reduce=False)))
        # print("Original log prob:", tf.reduce_sum(log_prob))
        # print("Penalized log prob:", pen_log_prob)
        if penalize:
            return pen_log_prob
        else:
            return log_prob
        
class SVPGPR(gpflow.models.SVGP):
    def __init__(self, X, Y, kernel, 
                 inducing_variable=None,
                 likelihood=gpflow.likelihoods.Gaussian(),
                 mean_function = None, num_inducing_points = 500,
                 noise_variance = 1.0, lam = 1.0, 
                 base_variances = None, gam = 1.0, 
                 **kwargs):
        
        # Set inducing variables if needed
        fix_inducing_points = False
        if inducing_variable is None:
            if num_inducing_points >= len(X):
                fix_inducing_points = True
                inducing_variable = gpflow.inducing_variables.InducingPoints(X)
            else:
                sample_points = np.random.choice(len(X), num_inducing_points)
                inducing_variable = gpflow.inducing_variables.InducingPoints(X[sample_points, :])

        super().__init__(
            kernel=kernel, 
            likelihood=likelihood, 
            inducing_variable=inducing_variable, 
            mean_function=mean_function,
            **kwargs 
        )
        self.lam = lam
        self.gam = gam        
        self.data = (
            tf.convert_to_tensor(X, dtype=tf.float64),
            tf.convert_to_tensor(Y, dtype=tf.float64)
        )

        # Set inducing points
        if fix_inducing_points is True:
            gpflow.utilities.set_trainable(self.inducing_variable, False)
        
        # Set base variances to ones if none given
        self.base_variances = base_variances
        # if base_variances is None:
        #     num_vars = len(find_variance_components(kernel, sum_reduce=False))
        #     self.base_variances = np.ones(shape=num_vars)
        # else:
        #     self.base_variances = base_variances

    def set_lambda(self, new_lam):
        self.lam = new_lam
        
    def set_gamma(self, new_gam):
        self.gam = new_gam
    
    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(X, f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        
        if self.base_variances is None:
            pen_factor = (tf.reduce_sum((len(X)) * self.lam *
                          find_variance_components(self.kernel, sum_reduce=False)))
        else:
            pen_factor = (tf.reduce_sum((len(X)) * self.lam *
                          (1/(self.base_variances)**self.gam) * 
                          find_variance_components(self.kernel, sum_reduce=False)))

        return tf.reduce_sum(var_exp) * scale - kl - pen_factor
    

def find_variance_components(kern, sum_reduce=True, penalize_factor_prod=1):
    """Retrieve the variance parameter of all kernel components recursively."""
    # print(kern.name)
    if kern.name == "sum":
        var_list = tf.stack(
            [find_variance_components(kern=x, sum_reduce=sum_reduce) for x in kern.kernels]
        )
        if sum_reduce:
            return tf.reduce_sum(var_list)
        else:
            return var_list
    elif kern.name == "product":
        return penalize_factor_prod*tf.reduce_prod([find_variance_components(x, sum_reduce) for x in kern.kernels])
    elif kern.name == "linear_coregionalization":

        # Calculate the weighted kernel components for each output
        # temp_weights = tf.matmul(
        #     a=kern.W,
        #     b=tf.reshape(
        #         tf.convert_to_tensor(
        #             [find_variance_components(x) for x in kern.kernels]
        #         ),
        #         shape=(-1, 1)
        #     )
        # )

        temp_weights = kern.W

        # return tf.matmul(a=temp_weights, b=temp_weights, transpose_a=True)
        if sum_reduce:
            return tf.reduce_sum(tf.abs(temp_weights))
        else:
            return tf.abs(temp_weights)
    else:
        if kern.name == "periodic":
            return tf.convert_to_tensor(kern.base_kernel.variance)
        else:
            return tf.convert_to_tensor(kern.variance)

def lasso_kernel_build(cat_vars = [], num_vars = [],
                       var_names = None,
                       second_order_numeric = False,
                       return_sum = False,
                       kerns = [gpflow.kernels.SquaredExponential()]):

    # Get a list of all variable/kernel combinations
    kernel_list = []

    if var_names is not None:
        var_list = []

    # Specify all the categorical options
    for c in cat_vars:
        # print(f"Adding kernel: categorical[{c}]")
        kernel_list += [Categorical(active_dims=[c])]

        if var_names is not None:
            var_list += ["categorical["+var_names[c]+"]"]

    # Same for numeric variables and kernel combinations
    for n in num_vars:
        for k in kerns:
            # print(f"Adding kernel: {k.name}[{n}]")
            k_copy = gpflow.utilities.deepcopy(k)
            k_copy.active_dims = [n]
            kernel_list += [k_copy]

            if var_names is not None:
                var_list += [k_copy.name + "[" + var_names[n] + "]"]

    # Now do interactions
    kern_len = len(kernel_list)

    # Do we want to interact all variables?
    if second_order_numeric is True:
        for k1 in range(kern_len):
        # for k1 in range(kern_len, len(cat_vars)): # What about if we only did interactions for categoricals?
            for k2 in range(k1, kern_len):
                # Skip categorical polys if same variable
                if k1 == k2 and k1 < len(cat_vars):
                    continue
                else:
                    kernel1 = gpflow.utilities.deepcopy(kernel_list[k1])
                    kernel2 = gpflow.utilities.deepcopy(kernel_list[k2])
                    # print(f"Adding kernel: {kernel1.name}[{kernel1.active_dims[0]}]*{kernel2.name}[{kernel2.active_dims[0]}]")
                    kernel_list += [kernel1*kernel2]

                    if var_names is not None:
                        var_list += [var_list[k1] + "*" + var_list[k2]]

    # We just want to interact categorical with numeric 
    else:
        for c in range(len(cat_vars)):
            for n in range(len(num_vars)):
                    kernel1 = gpflow.utilities.deepcopy(kernel_list[c])
                    kernel2 = gpflow.utilities.deepcopy(kernel_list[len(cat_vars)+n])
                    # print(f"Adding kernel: {kernel1.name}[{kernel1.active_dims[0]}]*{kernel2.name}[{kernel2.active_dims[0]}]")
                    kernel_list += [kernel1*kernel2]

                    if var_names is not None:
                        var_list += [kernel1.name + "[" + var_names[c] + "]*" + kernel2.name + "[" + var_names[len(cat_vars)+n] + "]"]

    if return_sum is True:
        out_kernel = gpflow.kernels.Sum(kernel_list)
    else:
        out_kernel = kernel_list

    if var_names is not None:
        return out_kernel, var_list
    else:
        return out_kernel

# Test case
# comp_lasso_kernel = lasso_kernel_build(cat_vars=[0,1,2], num_vars=[3,4])

def parallel_fold_test(X, Y, k, lam, gam, base_variances, 
                       f_val, num_inducing_points,
                       freeze_inducing, freeze_variances,
                       max_iter=50000,
                       verbose=False, likelihood="gaussian",
                       lasso=True, keep_data=True):
    # Fit each model with different fold
    # Optimize hyperparameters
    temp_m, temp_bic = kernel_test_reg(
        X=np.delete(X, f_val, axis=0),
        Y=np.delete(Y, f_val, axis=0),
        k=k,
        lasso=lasso,
        lam=lam,
        gam=gam,
        base_variances=base_variances,
        max_iter=max_iter,
        keep_data=keep_data,
        num_inducing_points=num_inducing_points,
        freeze_inducing=freeze_inducing,
        freeze_variances=freeze_variances,
        verbose=verbose,
        likelihood=likelihood
    )

    # Store holdout loglik (if no model could be fit then return worst-case)
    if temp_m is None:
        return temp_m, np.nan
    else:
        log_lik = np.mean(temp_m.predict_log_density(data=(X[f_val], Y[f_val])))
        return temp_m, log_lik
        # y_hat, cov_hat = temp_m.predict_f(X[f_val])
        # mse = np.mean((Y[f_val] - y_hat)**2)
        # return temp_m, mse
    
def kernel_test_reg(X, Y, k, num_restarts=5, random_init=True,
                verbose=False, likelihood='gaussian',
                lasso=False, lam=0, gam=0,
                base_variances=None, max_iter=50000,
                use_priors=True, keep_data=False,
                X_holdout=None, Y_holdout=None, split=False,
                freeze_inducing=False, freeze_variances=False,
                random_seed=None,
                **kwargs):
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
    better_model_seen = False

    for i in range(num_restarts):
        
        # # Check to see if it worthwhile to keep restarting
        # if better_model_seen is False and i > ceil((i+1)/2):
        #     if verbose:
        #         print(f'Tried {i+1} restarts and nothing better seen, stopping!')
        #     break
        
        # Specify model
        if lasso:

            # Go from string to gpflow likelihood object
            if type(likelihood) == str:
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

            if "num_inducing_points" not in kwargs:
                kwargs["num_inducing_points"] = 500
            elif kwargs["num_inducing_points"] > X.shape[0]:
                kwargs["num_inducing_points"] = X.shape[0]

            # Break out the number of latent GPs
            # print(k)
            if hasattr(k, "W"):
                # print(f"W shape = {k.W.shape}")
                # print(f"Y shape = {Y.shape}")
                num_latent = k.W.shape[1]
            else:
                num_latent = 1

            m = SVPGPR(
                X=X,
                Y=Y,
                kernel=k,
                lam=lam,
                gam=gam,
                base_variances=base_variances,
                likelihood=gp_likelihood,
                # num_inducing_points=kwargs["num_inducing_points"]
                inducing_variable=gpflow.inducing_variables.SeparateIndependentInducingVariables(
                    [gpflow.inducing_variables.InducingPoints(
                        # X_[:kwargs["num_inducing_points"],:]
                        X_[np.random.randint(low=0, high=X.shape[0], size=kwargs["num_inducing_points"]), :]
                        ) 
                        for X_ in [X.copy() for _ in range(num_latent)]
                    ]
                ),
                # inducing_variable=gpflow.inducing_variables.SharedIndependentInducingVariables(
                #     gpflow.inducing_variables.InducingPoints(
                #         X[np.random.randint(low=0, high=X.shape[0], size=kwargs["num_inducing_points"]), :].copy()
                #     )
                # ),
                num_latent_gps=num_latent
            )
        elif likelihood == 'gaussian':
            m = gpflow.models.GPR(
                #             m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k)#+gpflow.kernels.Constant())#,
            #mean_function=gpflow.mean_functions.Constant())#,
        #                 likelihood=gpflow.likelihoods.Gaussian())
        elif likelihood == 'exponential':
            m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k,
                # mean_function=gpflow.mean_functions.Constant(),
                likelihood=gpflow.likelihoods.Exponential())
        elif likelihood == 'poisson':
            m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k,
                # mean_function=gpflow.mean_functions.Constant(),
                likelihood=gpflow.likelihoods.Poisson())
        elif likelihood == 'gamma':
            m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k,
                # mean_function=gpflow.mean_functions.Constant(),
                likelihood=gpflow.likelihoods.Gamma())
        elif likelihood == 'bernoulli':
            m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k,#+gpflow.kernels.Constant(),
                # mean_function=gpflow.mean_functions.Constant(),
                likelihood=gpflow.likelihoods.Bernoulli())
        else:
            print('Unknown likelihood requested.')

        # Freeze requested model parameters from training
        if freeze_inducing is True:
            gpflow.utilities.set_trainable(m.inducing_variable.inducing_variables, False)

        if freeze_variances is True:
            freeze_variance_parameters(m.kernel)

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
            for param_name, param_val in gpflow.utilities.parameter_dict(m).items():
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
                    p.assign(
                        p.transform_fn(unconstrain_vals)
                    )
            
            for p in m.likelihood.trainable_parameters:
                if len(p.shape) <= 1:
                    unconstrain_vals = np.random.normal(
                        size=p.numpy().size
                        ).reshape(p.numpy().shape)
                    p.assign(
                        p.transform_fn(unconstrain_vals)
                    )
            
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
                    options={'maxiter': max_iter}
                )
            else:
                opt_res = gpflow.optimizers.Scipy().minimize(
                    m.training_loss,
                    m.trainable_variables,
                    # method="Nelder-Mead",
                    options={'maxiter': max_iter}
                )
            # adam_opt_params(m)
            # scipy_opt_params(m)

            # Check if model converged
            if opt_res["success"] is False:
                if verbose:
                    print("Warning: optimizer did not converge!")

        except Exception as e:
            if verbose:
                print(f'Optimization not successful, skipping. Error: {e}')
                print(i)
            if best_model == None and i == num_restarts - 1:
                return m, -1*best_loglik
            continue

        #         print(opt_results)
        # Now check to see if this is invertible
        try:
            m_, v_ = m.predict_y(m.data[0])
        except Exception as e:
            if verbose:
                print('Covariance matrix not invertible, removing model.')
            # If not invertible then revert back to best model, unless last try
            if best_model == None and i == num_restarts - 1:
                return best_model, -1*best_loglik
            else:
                m = best_model

        # Check if better values found and save if so
        #         if m.log_marginal_likelihood() > best_loglik:
        if m.name == "svpgpr":
            cur_loglik = m.log_posterior_density(data=(X,Y)).numpy()
        else:
            cur_loglik = m.log_posterior_density().numpy()
        if cur_loglik > best_loglik: 
            best_loglik = cur_loglik
            best_model = gpflow.utilities.deepcopy(m)
            better_model_seen = True
            if verbose:
                print(f'New best log likelihood: {best_loglik}')
        else: 
            del m

    #     # Set hyperparameters to best found values
    #     for l in range(len(m.trainable_parameters)):
    #         print(best_params[l])
    #         m.trainable_parameters[l].assign(best_params[l])

    # Return none and worst BIC if we can't fit a single
    if best_model == None:
        return best_model, -1*best_loglik

    # Calculate information criteria
    if split:
        yhat_holdout = best_model.predict_f(X_holdout)
        estimated_loglik = best_model.likelihood.predict_log_density(
            yhat_holdout[0],
            yhat_holdout[1],
            Y_holdout).numpy().sum()
        bic = round(-1*estimated_loglik, 2)
    else:
        estimated_loglik = best_loglik #best_model.log_posterior_density().numpy()

        bic = round(calc_bic(
            #         loglik=best_model.log_marginal_likelihood().numpy(),
            loglik=estimated_loglik,
            n=X.shape[0],
            k=len(best_model.trainable_parameters)),
            2)

    # Print out info if requested
    if verbose:
        print(f'Model: {print_kernel_names(k)}, BIC: {bic}')

    # Delete data from model object
    if not keep_data:
        best_model.data = None

    # Return fitted GP model and bic
    # Predictions
    #     print(best_model.predict_f(X))
    return best_model, bic

# Search over lambdas
def lam_search(kernel, X, Y,
               lam_list = None, num_lams=20, 
               gam_list = [0.], num_inducing_points=500,
               freeze_inducing = False, freeze_variances = False,
               k_fold=5, max_iter=50000,
            #    strata = None, 
               unit_col = None,
               likelihood="gaussian",
               max_jobs = -1, base_model=None,
               random_seed = None, verbose = False,
               return_all = False, early_stopping = True,
               fit_best = True):

    #TODO: Add strata option
    if return_all is True:
        model_dict = {}

    # Set seed if passed in
    if random_seed is not None: np.random.seed(random_seed)
    
    # Set base variances from base model
    if base_model is not None:
        base_variances = find_variance_components(
            base_model.kernel,
            sum_reduce=False
        )
    else:
        base_variances = None
        
    # Set lambda range if none given
    if lam_list is None:
        if verbose:
            print('Finding best lambda range now')
            
        # if base_model is None:
        #     base_model, _ = kernel_test_reg(
        #         X=X,
        #         Y=Y,
        #         k=gpflow.utilities.deepcopy(model_spec.kernel),
        #         keep_data=True
        #     )
        #     base_ll = base_model.log_marginal_likelihood()
        # else:
        #     base_ll = base_model.log_marginal_likelihood()
        # intercept_ll = norm(loc=0, scale=Y.std()).logpdf(Y).sum()
        
        # Now backtrack to find the max lambda that would produce
        # an intercept-only model
        # tf.reduce_sum(self.lam *
        #       (1/(self.base_variances)**gam_val) * 
        #       find_variance_components(base_model.kernel, sum_reduce=False)))
        
        # max_lambda = abs(base_ll*intercept_ll)
        max_lambda = Y.var()
        # lam_list = np.log(np.linspace(start=0, stop=max_lambda, num=num_lams))
        lam_list = np.insert(
            np.exp(
                np.linspace(
                    start=-10, 
                    stop=np.log(max_lambda), 
                    num=num_lams-1
                )
            ),
            0,
            0
        ).round(5)

    # Set up our k-fold set
    # Sample at either the unit or observation level
    if unit_col is None:
        sample_idx = np.arange(0, X.shape[0])
    else:
        sample_idx = np.unique(X[:, unit_col])

        # Check to see if there are enough people to have # of folds
        assert len(sample_idx) >= k_fold, (
            'Not enough unique units for number of folds requested, '
            f'{len(sample_idx)} unit(s) < {k_fold} fold(s)'
            )

    # Shuffle set
    np.random.shuffle(sample_idx)

    # Break up into folds
    div, mod = divmod(len(sample_idx), k_fold)
    folds = [
        sample_idx[i * div + min(i, mod) : (i + 1) * div + min(i + 1, mod)]
        for i in range(k_fold)
    ]

    # If the sample_idx is as the unit level we need to get row level
    if unit_col is not None:
        folds = [np.where(np.in1d(X[:, unit_col], f))[0] for f in folds]

    # print(f"folds = {folds}")

    # Set up objects to hold output
    val_log_lik = {key: {gam_key: [] for gam_key in gam_list} for key in lam_list}
    best_lam = None
    best_gam = None
    best_log_lik = None
    best_se = None
    stop_now = False
    # len(lam_list) * [np.nan]
    
    output_frame = pd.DataFrame()
    
    # Fit each model with a given lambda level
    for l, l_val in enumerate(lam_list):
        
        if stop_now is True:
            break
        
        for g, g_val in enumerate(gam_list):
            if verbose:
                print(f"lambda value = {l_val}, gamma value = {g_val}")

            # Fit each fold in parallel
            if max_jobs == -1:
                max_jobs = len(folds)
            par_res = Parallel(n_jobs=max_jobs)(delayed(parallel_fold_test)(
                X=X,
                Y=Y,
                k=gpflow.utilities.deepcopy(kernel),
                lam=l_val,
                f_val=f_val,
                gam=g_val,
                base_variances=base_variances,
                max_iter=max_iter,
                num_inducing_points=num_inducing_points,
                freeze_inducing=freeze_inducing,
                freeze_variances=freeze_variances,
                verbose=verbose,
                likelihood=likelihood
            ) for f_val in folds)
            
            val_log_lik[l_val][g_val] = [x[1] for x in par_res]
            
            if return_all is True:
                # model_dict[l_val] = [cut_kernel_components(x[0]) for x in par_res]
                model_dict[l_val] = [x[0] for x in par_res]

            # See if mean cross-validation log likelihood is better than current best
            # if best_log_lik is None or best_log_lik <= np.mean(val_log_lik[l_val][g_val]):
            if best_log_lik is None or best_log_lik <= np.mean(val_log_lik[l_val][g_val]):
                best_lam = l_val
                best_gam = g_val
                best_se = (
                    np.std(val_log_lik[l_val][g_val])
                    /np.sqrt(k_fold)
                )
                best_log_lik = np.mean(val_log_lik[l_val][g_val])
                if verbose:
                    print(f"ll = {best_log_lik}, se = {best_se}")
                
            # Check to see if we have passed the best options already
            if early_stopping:
                if np.mean(val_log_lik[l_val][g_val]) < (best_log_lik - 1.96*best_se):
                    if verbose:
                        print("Stopping early!")
                        print(np.mean(val_log_lik[l_val][g_val]))
                        print((best_log_lik - 1.96*best_se))
                    stop_now = True

    # Prepare output dictionary
    out = {}
    out["cv_log_lik"] = val_log_lik
    out["best_lambda"] = best_lam
    out["best_gamma"] = best_gam

    # Should we also return the best model fit on the entire dataset?
    if fit_best is True:
        best_m, best_bic = kernel_test_reg(
            X=X,
            Y=Y,
            k=gpflow.utilities.deepcopy(kernel),
            lasso=True,
            lam=best_lam,
            gam=best_gam,
            base_variances=base_variances,
            max_iter=max_iter,
            keep_data=True,
            num_inducing_points=num_inducing_points,
            freeze_inducing=freeze_inducing,
            freeze_variances=freeze_variances,
            verbose=verbose,
            likelihood=likelihood
        )
        out["final_model"] = best_m

    if return_all is True:
        out["model_list"] = model_dict
    
    return out


def cut_kernel_components(model, var_cutoff=0.001):
    
    if model is None:
        return model
    
    # Get variance components for each additive kernel part
    var_parts = find_variance_components(model.kernel, sum_reduce=False)
    var_flag = np.argwhere(var_parts >= var_cutoff).transpose().tolist()[0]

    # Copy model
    new_model = gpflow.utilities.deepcopy(model)

    # Figure out which ones should be kept
    new_model.kernel = gpflow.kernels.Sum([model.kernel.kernels[i] for i in var_flag])

    # Also make sure we only keep base variances that remain
    new_model.base_variances = model.base_variances[var_flag]

    return new_model
