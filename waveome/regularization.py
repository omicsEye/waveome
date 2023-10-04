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
from .model_fitting import kernel_test_reg
from .model_types_DEPR import PGPR, SVPGPR
from .utilities import (
    calc_bic,
    print_kernel_names,
    find_variance_components,
    freeze_variance_parameters,
    gp_likelihood_crosswalk,
    f64
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def full_kernel_build(cat_vars = [], num_vars = [],
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
    

def make_folds(X, unit_col, k_fold = 5, random_seed=None):

    # Set seed if desired
    if random_seed is not None:
        np.random.seed(random_seed)
        
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

    return folds

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
               fit_best = True, prune_best=True):

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
        #         k=gpflow.utilities.deepcopy(kernel),
        #         keep_data=True
        #     )
        #     base_ll = base_model.log_marginal_likelihood()
        # else:
        #     base_ll = base_model.log_marginal_likelihood()
        # intercept_ll = norm(loc=0, scale=Y.std()).logpdf(Y).sum()
        
        # # Now backtrack to find the max lambda that would produce
        # # an intercept-only model
        # # tf.reduce_sum(self.lam *
        # #       (1/(self.base_variances)**gam_val) * 
        # #       find_variance_components(base_model.kernel, sum_reduce=False)))
        
        # max_lambda = abs(base_ll*intercept_ll)
        max_lambda = 2*Y.var()
        print(f"max lambda: {max_lambda}")
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
    folds = make_folds(X=X, unit_col=unit_col, k_fold=k_fold)

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

        # Prune model components
        best_m = cut_kernel_components(best_m)

        out["final_model"] = best_m

    if return_all is True:
        out["model_list"] = model_dict
    
    return out


def cut_kernel_components(
        model, 
        var_cutoff: float = 0.001
    ):
    ''' Prune out kernel components with small variance parameters
     
    Parameters
    ----------
    model
    var_cutoff

    Returns
    -------
    GPflow model
    '''

    # Return empty model object if none passed in
    if model is None:
        return model
    
    # Get variance components for each additive kernel part
    var_parts = find_variance_components(model.kernel, sum_reduce=False)
    var_flag = np.argwhere(var_parts >= var_cutoff).transpose().tolist()[0]

    # Copy model
    new_model = gpflow.utilities.deepcopy(model)

    # Figure out which ones should be kept
    new_model.kernel = gpflow.kernels.Sum(
        [model.kernel.kernels[i] for i in var_flag]
    )

    # Also make sure we only keep base variances that remain
    if hasattr(model, 'base_variances'):
        if model.base_variances is not None:
            new_model.base_variances = model.base_variances[var_flag]

    return new_model
