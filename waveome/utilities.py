import numpy as np
import gpflow
from gpflow.utilities import set_trainable #, ci_niter
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
f64 = gpflow.utilities.to_default_float

def calc_bic(loglik, n, k):
    # return k*np.log(n)-2*loglik
    return 2 * k - 2 * loglik

def coregion_freeze(k):
    """ Freeze parameters associated with coregion kernel, for individual level effets. """
    
    if k.name == 'coregion':
        #print('Found coregion kernel, freezing parameters.')
        k.W.assign(np.zeros_like(k.W))
        k.kappa.assign(np.ones_like(k.kappa))
        set_trainable(k.W, False)
        set_trainable(k.kappa, False)
#         set_trainable(k, False)

def coregion_search(kern_list):
    """ Search through GP kernel list to find coregion kernels. """
    
    for k in kern_list:
        if hasattr(k, 'kernels'):
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
    sse = np.sum((Y - Y_bar)**2)
    
    # Calculate overall model predictions
    mu_all_hat, var_all_hat = m.predict_y(X)
    ssr_total = np.sum((Y-mu_all_hat)**2)
    total_rsq = 1 - (ssr_total/sse)
    
    # For each kernel component gather predictions
    ssr_list = []
    k = m.kernel
    if k.name == 'sum':
        for k_idx in range(len(k.kernels)):
            # Break off kernel component
            # m_copy.kernel = k_sub
            # mu_hat, var_hat = m_copy.predict_y(X)
            mu_hat, var_hat, samps, cov_hat = individual_kernel_predictions(
                model=m,
                kernel_idx=k_idx,
                X=m.data[0]
            )
            ssr_list += [np.sum((mu_all_hat - mu_hat)**2)]
        
        for k_idx in range(len(k.kernels)):
            rsq += [np.round(total_rsq*(1 - ssr_list[k_idx]/sum(ssr_list)), 3)]
    else:
        mu_hat, var_hat = m_copy.predict_y(X)
        ssr = np.sum((mu_all_hat - mu_hat)**2)
        rsq += [np.round(total_rsq, 3)]
        
    # Gather the final bit for noise
    # rsq += [np.round(1 - sum(rsq),3)]
    rsq += [np.round(1 - total_rsq, 3)]

    return rsq

def calc_residuals(m):
    """
    Calculate pearson residuals from model
    """

    # Get observed predictions and variance
    mean_y, var_y = m.predict_y(m.data[0])
    
    # Calculate standardized residuals
    resids = ((tf.cast(m.data[1], tf.float64) - mean_y)/np.sqrt(var_y)).numpy()
    
    return resids

def calc_bhattacharyya_dist(model1, model2, X):
    """
    Calculate the Bhattacharyya distance between two resulting MVNormal distributions.
    """
    
    # Calculate means and variances
    mu1, var1 = model1.predict_f(X)
    mu2, var2 = model2.predict_f(X)
    
    # Also calculate covariance matrices
    # Pull kernel covariance matrices
    cov1 = model1.kernel.K(X)
    cov2 = model2.kernel.K(X)
    
    # Then add likelihood noise if necessary
    if model1.name == 'gpr' and model2.name == 'gpr':
        cov1 += tf.linalg.diag(tf.repeat(model1.likelihood.variance, X.shape[0]))
        cov2 += tf.linalg.diag(tf.repeat(model2.likelihood.variance, X.shape[0]))
    
    # Calculate average sigma
    cov_all = (cov1 + cov2)/2.
    
    # After that calculate closed form of Bhattacharyya distance
    dist_b = (#(1/8.) * tf.transpose(mu1 - mu2)@tf.linalg.inv(cov_all)@(mu1 - mu2) + 
              0.5 * np.log(tf.linalg.det(cov_all)/(np.sqrt(tf.linalg.det(cov1)*tf.linalg.det(cov2)))))
    
    return dist_b

def replace_kernel_variables(k_name, col_names):
    """
    Takes in indexed kernel names and original column names, then replaces and spits out
    new string.
    """
    
    # Make copy of kernel name
    new_k_name = k_name
    
    for i, c in enumerate(col_names):
        new_k_name = new_k_name.replace('['+str(i)+']', '['+c+']')
        
    return new_k_name

def check_if_model_exists(model_name, model_list):
    """
    Checks if current model name is in list of fit models.
    """
    found_model = None
    
    # First split models into additive components
    model_name_split = model_name.split('+')
    model_list_split = [x.split('+') for x in model_list]
    
    # Then order the resulting product pieces
    model_name_split_ordered = [''.join(sorted(x)) for x in model_name_split]
    model_list_split_ordered = [''.join(sorted(x)) for y in model_list_split for x in y]
            
    term_diff = [set(model_name_split_ordered) ^ 
                     set([''.join(sorted(x)) for x in y]) 
                 for y in model_list_split]
    
    if set() in term_diff:
        found_model = True
    else:
        found_model = False
    
    return found_model

def hmc_sampling(model, burn_in=500, samples=1000, random_seed=None,
                 step_size=0.01, accept_prob=0.80):
    
    model = gpflow.utilities.deepcopy(model)
    
    # Set priors if they don't already have them
    for p in model.parameters:
        if p.prior == None:
            p.prior = tfd.Gamma(f64(2), f64(2))
    
    # Set helper
    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density,
        model.trainable_parameters
    )
    
    # Set HMC options
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn,
        num_leapfrog_steps=10,
        step_size=step_size
    )
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc,
        num_adaptation_steps=10,
        target_accept_prob=f64(accept_prob),
        adaptation_rate=0.1
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
    
    return {'samples': strain_samples, 
            'unconstrained_samples': samples, 
            'traces': traces}

def print_kernel_names(kernel, with_idx=False):
    names = []
    
    if hasattr(kernel, 'kernels') == False:
        if with_idx:
            return kernel.name + "[" + str(kernel.active_dims[0]) + "]"
        else:
            return kernel.name
    elif kernel.name == 'sum':
        return [print_kernel_names(x, with_idx) for x in kernel.kernels]
        # names.append('+'.join([print_kernel_names(x) for x in kernel.kernels]))
    elif kernel.name == 'product':
        return '*'.join([print_kernel_names(x, with_idx) for x in kernel.kernels])
    return names

def adam_opt_params(m, iterations=500, eps=0.1):
    prev_loss = np.Inf
    for i in range(iterations):
        tf.optimizers.Adam(learning_rate=0.1, epsilon=0.1).minimize(
            m.training_loss, m.trainable_variables
        )
        
        if abs(prev_loss - m.training_loss()) < eps:
            break
        else:
            prev_loss = m.training_loss()
    return None

def variance_contributions(m, k_names, lik='gaussian'):
    """
    Takes a GP model and returns the percent of variance explained for each
    additive component. 
    """
    
    variance_list = []
    
    # Split kernel into additive pieces
    kernel_names = k_names.split('+')
    
    # Check if there is only one kernel component, otherwise go through all
    if len(kernel_names) == 1:
        if m.kernel.name == 'product':
            prod_var = 1
            for k in m.kernel.kernels:
                if k.name == 'periodic':
                    prod_var *= k.base_kernel.variance.numpy().round(3)
                else:
                    prod_var *= k.variance.numpy().round(3)
            variance_list += [prod_var.tolist()]
        
        elif m.kernel.name == 'sum':
            sum_var = 0
            for k in m.kernel.kernels:
                if k.name == 'periodic':
                    sum_var += k.base_kernel.variance.numpy().round(3)
                else:
                    sum_var += k.variance.numpy().round(3)
            variance_list += [sum_var.tolist()]
            
        elif m.kernel.name == 'periodic':
            variance_list += [m.kernel.base_kernel.variance.numpy().round(3)]
        else:
            variance_list += [m.kernel.variance.numpy().round(3)]
    else:
        for k in range(len(kernel_names)):
            if m.kernel.kernels[k].name == 'product':
                prod_var = 1
                for k2 in m.kernel.kernels[k].kernels:
                    if k2.name == 'periodic':
                        prod_var *= k2.base_kernel.variance.numpy().round(3)
                    else:
                        prod_var *= k2.variance.numpy().round(3)
                variance_list += [prod_var.tolist()]
                
            elif m.kernel.kernels[k].name == 'sum':
                sum_var = 0
                for k2 in m.kernel.kernels[k].kernels:
                    if k2.name == 'periodic':
                        sum_var += k2.base_kernel.variance.numpy().round(3)
                    else:
                        sum_var += k2.variance.numpy().round(3)
                variance_list += [sum_var.tolist()]
                    
            elif m.kernel.kernels[k].name == 'periodic':
                variance_list += [m.kernel.kernels[k].base_kernel.variance.numpy().round(3).tolist()]
                
            else:
                variance_list += [m.kernel.kernels[k].variance.numpy().round(3).tolist()]
        
    # Get likelihood variance
    if lik == 'gaussian':
        variance_list += [m.likelihood.variance.numpy().round(3).tolist()]
    else:
        variance_list += [np.std(calc_residuals(m))**2]
#     elif lik == 'exponential':
#     elif lik == 'poisson':
#     elif lik == 'gamma':
#     elif lik == 'bernoulli':
#         variance_list += 
#     else:
#         raise ValueError('Unknown likelihood function specified.')
    return variance_list

def variance_contributions_diag(m, lik='gaussian'):
    
    variance_list = []
    k = m.kernel
    
    # Extract variance from kernel components
    if k.name == 'sum':
        for i in range(len(k.kernels)):
            mu_, var_, samps_, cov_ = individual_kernel_predictions(
                model=m,
                kernel_idx=i,
                X=m.data[0])
        # for k_sub in k.kernels:
            # variance_list += [np.mean(k_sub.K_diag(m.data[0]))]
    elif k.name == 'product':
        temp_prod = np.ones_like(m.data[0][:,0])
        for k_sub in k.kernels:
            temp_prod *= k_sub.K_diag(m.data[0])
        variance_list += [np.mean(temp_prod)]
    else:
        variance_list += [np.mean(k.K_diag(m.data[0]))]
        
    # Extract variance from likelihood function
    if lik == 'gaussian':
        variance_list += [m.likelihood.variance.numpy().round(3).tolist()]
    else:
        variance_list += [np.std(calc_residuals(m))**2]
    return variance_list

def individual_kernel_predictions(
        model, 
        kernel_idx, 
        X=None, 
        white_noise_amt=1e-6,
        predict_y=False,
        num_samples=10
    ):
    """Predict contribution from individual kernel component.
    
    Parameters
    ----------
    model : gpflow.model
    
    kernel_idx : Integer
    
    X : Numpy array for prediction points
    
    white_noise_amt : Float
                      Amount of diagonal noise to add to covariance matricies
    
    predict_y : Boolean
                Add Gaussian noise from likelihood function?
                       
    num_samples : Integer
                  Number of samples to draw from the posterior component
        
    Attributes
    ----------
    
    """
    
    # conditional = gpflow.conditionals.base_conditional
    # f_mean_zero, f_var = conditional(
    #     kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
    # ) 
    
    # Build each part of the covariance matrix
    if model.kernel.name == "sum":
        sigma_21 = tf.cast(model.kernel.kernels[kernel_idx].K(X=model.data[0], X2=X), tf.float64)
        sigma_11 = tf.cast(model.kernel.kernels[kernel_idx].K(X=X), tf.float64)
    else:
        sigma_21 = tf.cast(model.kernel.K(X=model.data[0], X2=X), tf.float64)
        sigma_11 = tf.cast(model.kernel.K(X=X), tf.float64)
        
    sigma_22 = tf.cast(model.kernel.K(X=model.data[0]),tf.float64)
    sigma_12 = tf.transpose(sigma_21)
        
    # Add likelihood noise if requested - otherwise small constant for invertibility
    if predict_y:
        sigma_22 += tf.linalg.diag(tf.repeat(model.likelihood.variance, model.data[0].shape[0]))
        sigma_11 += tf.linalg.diag(tf.repeat(model.likelihood.variance, X.shape[0]))
    else: # Was adding 1e-4 noise, might not need that though 
        sigma_22 += tf.linalg.diag(tf.repeat(f64(white_noise_amt), model.data[0].shape[0]))
        sigma_11 += tf.linalg.diag(tf.repeat(f64(white_noise_amt), X.shape[0]))

    # Now put all of the pieces together into one matrix
    sigma_full = tf.concat([tf.concat(values=[sigma_11, sigma_12], axis=1), 
                            tf.concat(values=[sigma_21, sigma_22], axis=1)], 
                           axis=0)
    
    # Invert sigma_22
    # Try LU decomposition first
    try:
        inv_sigma_22 = tfp.math.lu_matrix_inverse(*tf.linalg.lu(sigma_22))
    except:
        print("Warning - Approximating the covariance inverse")
        inv_sigma_22 = tf.linalg.pinv(sigma_22)
        
    # Now calculate mean and variance
    pred_mu = (np.zeros((X.shape[0], 1)) + 
               tf.matmul(a=tf.matmul(a=sigma_12, #b=tf.linalg.inv(sigma_22)), 
                                     b=inv_sigma_22),
                         b=(model.data[1] - np.zeros((model.data[0].shape[0], 1)))))

    # Covariance function
    pred_cov = (sigma_11 - tf.matmul(a=sigma_12,
                                     b=tf.matmul(a=inv_sigma_22, 
                                                 #a=tf.linalg.inv(sigma_22),
                                                 b=sigma_21)))
    # Variance component
    pred_var = tf.linalg.diag_part(pred_cov)

    # Also pull some function samples
    # posterior_dist = tfp.distributions.MultivariateNormalFullCovariance(
    #     loc=tf.transpose(pred_mu),
    #     covariance_matrix=pred_cov,
    #     )
    # Need to update this to silence tensorflow warning
    posterior_dist = tfp.distributions.MultivariateNormalTriL(
        loc=tf.transpose(pred_mu),
        scale_tril=tf.linalg.cholesky(pred_cov)
    )
    sample_fns = posterior_dist.sample(sample_shape=num_samples)
    sample_fns = tf.transpose(tf.reshape(sample_fns, (num_samples, -1)))
    
    return pred_mu, pred_var, sample_fns, pred_cov
