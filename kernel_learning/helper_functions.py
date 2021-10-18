# Libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import gpflow
from gpflow.utilities import set_trainable
import re
import contextlib
import joblib
from tqdm import tqdm    
from joblib import Parallel, delayed

# Classes
class Categorical(gpflow.kernels.Kernel):
    def __init__(self, active_dims):
        super().__init__(active_dims=active_dims)
        self.variance = gpflow.Parameter(
            1.0, 
            transform=gpflow.utilities.positive()
        )
#         self.rho = gpflow.Parameter(
#             1.0, 
#             transform=gpflow.utilities.positive()
#         )
    
    #@tf.autograph.experimental.do_not_convert
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return (self.variance * #self.rho * 
                tf.cast(tf.equal(X, tf.reshape(X2,[1,-1])),
                        tf.float64))
    
    def K_diag(self, X):
#         return self.variance #* tf.reshape(X, (-1,))
        return self.variance * tf.reshape(tf.ones_like(X), (-1,)) + 1e-6

# Helper functions
def gp_predict_fun(gp, #X, Y, x_min, x_max, 
                   x_idx, unit_idx, 
                   unit_label=1, num_funs=10):  
    """
    Plot marginal closed-form posterior distribution.
    """
    
    # Pull training data from model
    X_train, Y_train = gp.data
    X_train = X_train.numpy()
    Y_train = Y_train.numpy()
    
    # Create test points
#     x_new = np.zeros_like(X)
#     x_new[:,x_idx] = np.linspace(x_min, x_max, X.shape[0])
#     x_new[:, unit_idx] = unit_label
    x_new = np.zeros((1000, X_train.shape[1]))
    x_new[:, x_idx] = np.linspace(X_train[:, x_idx].min(),
                                  X_train[:, x_idx].max(),
                                  1000)
    x_new[:, unit_idx] = unit_label

    # Predict mean and variance on new data
#     mean, var = gp.predict_f(x_new)
    mean, var = gp.predict_y(x_new)


    # Pull some posterior functions
    tf.random.set_seed(1) 
    samples = gp.predict_f_samples(x_new, num_funs) 

    # Generate plot
#     p = plt.figure(figsize=(10, 5))
    fig, ax = plt.subplots(figsize=(10,5))
#     p = sns.scatterplot(x=X[:,x_idx],
#                     y=Y.flatten(),
#                     hue=X[:,unit_idx].astype(int).astype(str),
#                         legend=False)
    person_rows = X_train[:,unit_idx] == unit_label
    p = sns.scatterplot(x=X_train[person_rows,x_idx],
                    y=Y_train.flatten()[person_rows],
#                     hue=X[:,unit_idx].astype(int).astype(str),
#                         legend=False
                       s=100,
                       color='black')
    p = sns.scatterplot(x=X_train[~np.array(person_rows),x_idx],
                    y=Y_train.flatten()[~np.array(person_rows)],
#                     hue=X[:,unit_idx].astype(int).astype(str),
#                         legend=False
                       s=10,
                       color='grey')
    p = sns.lineplot(x=x_new[:,x_idx],
#                  y=mean.numpy().flatten(),
                 y=gp.likelihood.invlink(mean.numpy().flatten()),
                 linewidth=2,
                    color='darkgreen')
    p.fill_between(
        x_new[:, x_idx],
        gp.likelihood.invlink(mean[:, 0] - 1.96 * np.sqrt(var[:, 0])),
        gp.likelihood.invlink(mean[:, 0] + 1.96 * np.sqrt(var[:, 0])),
        color='lightgreen',
        alpha=0.5,
    )
    p.plot(x_new[:,x_idx], 
           gp.likelihood.invlink(samples[:, :, 0].numpy().T),# "C0", 
           color='dimgray',
           linewidth=0.5)
#     plt.close()
    return(p)

def calc_bic(loglik, n, k):
    return k*np.log(n)-2*loglik

# Kernel search helper functions
# def coregion_freeze(k):
#     """ Freeze parameters associated with coregion kernel, for individual level effets. """
    
#     if k.name == 'coregion':
#         #print('Found coregion kernel, freezing parameters.')
#         k.W.assign(np.zeros_like(k.W))
#         k.kappa.assign(np.ones_like(k.kappa))
#         set_trainable(k.W, False)
#         set_trainable(k.kappa, False)
# #         set_trainable(k, False)

# def coregion_search(kern_list):
#     """ Search through GP kernel list to find coregion kernels. """
    
#     for k in kern_list:
#         if hasattr(k, 'kernels'):
#             coregion_search(k.kernels)
#         else:
#             coregion_freeze(k)

def print_kernel_names(kernel):
    names = []
    if hasattr(kernel,'kernels')==False:
        return kernel.name
    for i in kernel.kernels:
        if i.name == 'sum':
            sub_names = print_kernel_names(i)
#             names.append('+'.join([x.name for x in i.kernels]))
            names.append('+'.join(sub_names))

        elif i.name == 'product':
            sub_names = print_kernel_names(i)
#             names.append('*'.join([x.name for x in i.kernels]))
            names.append('*'.join(sub_names))

        else:
            names.append(i.name)

    return(names)

def kernel_test(X, Y, k, num_restarts=3, random_init=False, verbose=False, likelihood='gaussian'):
    """
    This function evaluates a particular kernel selection on a set of data. 
    
    Inputs:
        X (array): design matrix
        Y (array): output matrix
        k (gpflow kernel): specified kernel
        
    Outputs:
        m (gpflow model): fitted GPflow model
    
    """
            
    # Randomize initial values for a number of restarts
    np.random.seed(9012)
    best_loglik = -np.Inf
    best_model = None
        
    for i in range(num_restarts):
        
        # Specify model
        if likelihood == 'gaussian':
#             m = gpflow.models.GPR(
            m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k,
                mean_function=gpflow.mean_functions.Constant(),
                likelihood=gpflow.likelihoods.Gaussian())
        elif likelihood == 'exponential':
            m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k,
                mean_function=gpflow.mean_functions.Constant(),
                likelihood=gpflow.likelihoods.Exponential())
        elif likelihood == 'poisson':
            m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k,
                mean_function=gpflow.mean_functions.Constant(),
                likelihood=gpflow.likelihoods.Poisson())
        elif likelihood == 'gamma':
            m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k,
                mean_function=gpflow.mean_functions.Constant(),
                likelihood=gpflow.likelihoods.Gamma())
        elif likelihood == 'bernoulli':
            m = gpflow.models.VGP(
                data=(X, Y),
                kernel=k,
                mean_function=gpflow.mean_functions.Constant(),
                likelihood=gpflow.likelihoods.Bernoulli())
        else:
            print('Unknown likelihood requested.')
        
        # Randomize initial values if not trained already
        if random_init:
            for p in m.trainable_parameters:
                if p.numpy() == 1:
                    p.assign(
                        np.random.uniform(
                            size=p.numpy().size
                        ).reshape(p.numpy().shape)
                    )
        
        # Optimization step for hyperparameters
        try:
            opt_results = gpflow.optimizers.Scipy().minimize(
                m.training_loss,
                m.trainable_variables)
        except:
            print('Optimization not successful, skipping.')
            return best_model, -1*best_loglik
        
#         print(opt_results)
        
        # Check if better values found and save if so
#         if m.log_marginal_likelihood() > best_loglik:
        if m.log_posterior_density() > best_loglik:
            best_loglik = m.log_posterior_density() #m.log_marginal_likelihood()
            best_model = m
            if verbose:
                print(f'New best log likelihood: {best_loglik.numpy()}')
            
#     # Set hyperparameters to best found values
#     for l in range(len(m.trainable_parameters)):
#         print(best_params[l])
#         m.trainable_parameters[l].assign(best_params[l])    

    # Calculate information criteria
    bic = round(calc_bic(
#         loglik=best_model.log_marginal_likelihood().numpy(),
        loglik=best_model.log_posterior_density().numpy(),
                   n=X.shape[0],
                   k=len(m.trainable_parameters)),
               2)
    
    # Print out info if requested
    if verbose:
        print(f'Model: {print_kernel_names(k)}, BIC: {bic}')
    
    # Return fitted GP model and bic
    # Predictions
#     print(best_model.predict_f(X))
    return best_model, bic

###############################################################################
# New functions
###############################################################################
def set_feature_kernels(f, kern_list, cat_vars):
    if f in cat_vars:
        k_list = [Categorical(active_dims=[f])]
    else:
        k_list = kern_list.copy()
        for k_ in k_list:
            k_.active_dims = np.array([f])
    return k_list

def loc_kernel_search(X, Y, kern_list, 
                      base_kern=None, base_name=None, 
                      cat_vars=[],
                      depth=0,
                      operation='sum',
                      prod_index=None,
                      lik='gaussian',
                      verbose=False):
    """
    This function performs the local kernel search.
    """
    
    bic_dict = {} 
    
    # Search over features in X
    for f in np.arange(X.shape[1]):
        if verbose:
            print(f'Working on feature {f} now')
            
        temp_kern_list = [gpflow.utilities.deepcopy(x) for x in kern_list]
        # Set kernel list based on feature currently searching
        k_list = set_feature_kernels(f=f,
                                     kern_list=temp_kern_list,
                                     cat_vars=cat_vars)
        # Add static kernel to test if first level and first feature
        if f==0 and depth==1:
            #print(f'Current list of kernels: {k_list}')
            k_list += [gpflow.kernels.Constant()]
            
        # Search over kernels
        for k in k_list:
            
            # Get kernel name and dimension
            k_info = k.name + str(k.active_dims)
            if verbose:
                print('Current kernel being tested: {}'.format(k_info))
            
            # Update kernel information with base if possible
            if base_kern != None:
#                 # Check if the new kernel is a copy of the previous categorical kernel
#                 # this would be redundant
#                 if k.name == 'categorical' and k_info in base_name:
#                     continue
                if operation == 'sum':
                    # Skip operation if categorical kernel exists
                    if 'categorical['+str(f)+']' not in base_name:
                        #print('Sum kernel being performed.')
                        try:
                            k = gpflow.kernels.Sum(kernels = [base_kern, k])
                            k_info = base_name + '+' + k_info
                            m, bic = kernel_test(X, Y, k, likelihood=lik)
                            #print(k_info)
                            bic_dict[k_info] = [k, m, bic, depth, base_name]
                        except:
                            None
                    
                elif operation == 'product':       
                    # Skip operation if categorical kernel exists
                    if 'categorical['+str(f)+']' not in base_name:
                        #print('Product kernel being performed.')
                        try:
                            # Set new variance to 1, don't train
#                             set_trainable(k.variance, False)
                            k = gpflow.kernels.Product(kernels = [base_kern, k])
                            k_info = base_name + '*' + k_info
                            m, bic = kernel_test(X, Y, k, likelihood=lik)
                            #print(k_info)
                            bic_dict[k_info] = [k, m, bic, depth, base_name]
                        except:
                            None
                    
                elif operation == 'split_product':
                    #print('Split product being tested.')
                    # Set new variance to 1, don't train
#                     set_trainable(k.variance, False)
                    new_dict = prod_kernel_creation(X=X,
                                                    Y=Y,
                                                    base_kernel=base_kern,
                                                    base_name=base_name,
                                                    new_kernel=k,
                                                    depth=depth,
                                                    lik=lik)
                    bic_dict.update(new_dict)
            else:
                m, bic = kernel_test(X, Y, k, likelihood=lik)
                bic_dict[k_info] = [k, m, bic, depth, 'None']
            
    return bic_dict

def prod_kernel_creation(X, Y, base_kernel, base_name, new_kernel, depth, lik):
    """
    Produce kernel for product operation
    """
    
    bic_dict = {}
    
    for feat in range(len(base_kernel.kernels)):
        temp_kernel = gpflow.utilities.deepcopy(base_kernel)
        temp_kernel.kernels[feat] = temp_kernel.kernels[feat] * new_kernel
        temp_name = base_name.split('+')
        k_info = new_kernel.name + str(new_kernel.active_dims)
        # Skip operation if categorical kernel exists
        #print('k_info:',k_info,'temp_name:',temp_name[feat])
        if 'categorical'+str(new_kernel.active_dims) not in temp_name[feat]:
            #print('Testing feature')
            try:
                temp_name[feat] = temp_name[feat] + '*' + k_info
                k_info = '+'.join(temp_name)

                m, bic = kernel_test(X, Y, temp_kernel, likelihood=lik)
                #print(k_info)
                bic_dict[k_info] = [temp_kernel, m, bic, depth, base_name]
            except:
                None
        
    return bic_dict

def check_if_better_metric(previous_dict, current_dict):
    """
    Check to see if a better metric was found in current layer, otherwise end search.
    """
    
    best_prev = min(x[2] for x in previous_dict.values())
    best_new = min(x[2] for x in current_dict.values())
    
    return True if best_new < best_prev else False

def keep_top_k(res_dict, depth, metric_diff = 6):
    
    best_bic = np.Inf
    out_dict = res_dict.copy()
    
    # Find results from the current depth
    best_bic = min([v[2] for k,v in res_dict.items() if v[3] == depth])
#     for k,v in res_dict.items():
#         if v[3] == depth and v[2] < best_bic:
    
    # Only keep results in diff window
    for k,v in res_dict.items():
        if v[3] == depth and (v[2]-best_bic) > metric_diff:
            out_dict.pop(k)
        
    return out_dict

def prune_best_model(res_dict, depth, lik):
    
    out_dict = res_dict.copy()
    
    # Get best model from current depth
    best_bic, best_model_name, best_model = min(
        [(i[2], k, i[1]) for k, i in res_dict.items()]
        )
#     print(f'Best model: {best_model_name}')
    
    # Split kernel name to sum pieces 
    kernel_names = best_model_name.split('+')
    
    # Loop through each kernel piece and prune out refit and evaluate
    if len(kernel_names) <= 1:
#         print('No compound terms to prune!')
        return res_dict
    
    for i in range(len(kernel_names)):
        k = gpflow.kernels.Sum(
            kernels = [k_ for i_, k_ in enumerate(best_model.kernel.kernels) 
                       if i_ != i])
        k_info = '+'.join([x_ for i_,x_ in enumerate(kernel_names) if i_ != i])
        m, bic = kernel_test(best_model.data[0], best_model.data[1], k, likelihood=lik)
        # If better model found then save it
        if bic < best_bic:
#             print(f'New better model found: {k_info}')
            out_dict[k_info] = [k, m, bic, depth, best_model_name]
    return out_dict

def full_kernel_search(X, Y, kern_list, cat_vars=[], max_depth=5, 
                       keep_all=False, early_stopping=True, prune=True,
                       lik='gaussian', verbose=False):
    """ 
    This function runs the entire kernel search, calling helpers along the way.
    """
    
    # Create initial dictionaries to insert
    search_dict = {}
    edge_list = []
    
    # Flag for missing values 
    x_idx = ~np.isnan(X).any(axis=1)
    y_idx = ~np.isnan(Y).flatten()
    
    # Get complete cases        
    comp_X = X[x_idx & y_idx]
    comp_y = Y[x_idx & y_idx]
    
    X = comp_X
    Y = comp_y
    
    # Go through each level of depth allowed
    for d in range(1, max_depth+1):
        if verbose:
            print(f'Working on depth {d} now')
        # If first level then there is no current dictionary to update
        if d == 1:
            search_dict = loc_kernel_search(
                X=X, 
                Y=Y, 
                kern_list=kern_list, 
                cat_vars=cat_vars,
                depth=d,
                lik=lik,
                verbose=verbose
            )
        else:
            # Create temporary dictionary to hold new results
            temp_dict = search_dict.copy()
            for k in search_dict.keys():
                # Make sure to only search through the previous depth
                if search_dict[k][3] != d-1:
                    continue
                
                cur_kern = search_dict[k][0]
                
                new_res = loc_kernel_search(
                    X=X, 
                    Y=Y, 
                    kern_list=kern_list,
                    base_kern=cur_kern,
                    base_name=k,
                    cat_vars=cat_vars,
                    depth=d,
                    lik=lik,
                    operation='sum',
                    verbose=verbose
                )
                temp_dict.update(new_res)
                
                # For product break up base kernel and push in each possibility
                # else just simple product with single term
                op = 'split_product' if cur_kern.name == 'sum' else 'product'
                new_res = loc_kernel_search(
                    X=X, 
                    Y=Y, 
                    kern_list=kern_list,
                    base_kern=cur_kern,
                    base_name=k,
                    cat_vars=cat_vars,
                    depth=d,
                    lik=lik,
                    operation=op,
                    verbose=verbose
                )
                temp_dict.update(new_res)
                
                # Update graph dictionary
                for k_ in new_res.keys():
                    edge_list += [(k, k_)]
            

            found_better = check_if_better_metric(search_dict, temp_dict)
            
            # Overwrite search dictionary with temporary results
            search_dict = temp_dict
            
            # Early stopping?
            if early_stopping:
                found_better = check_if_better_metric(search_dict, 
                                                      temp_dict)

                # If no better kernel found then exit search
                if not found_better:
                    if verbose:
                        print('No better kernel found in layer, exiting search!')
                    # Prune best model
                    if prune:
                        search_dict = prune_best_model(search_dict, depth=d, lik=lik)
                    break
                else:
                    None
#                     print('Found better kernel in next layer, would like to continue search.')
        
        # Filter out results from current depth to just keep best kernel options
        if not keep_all:
            search_dict = keep_top_k(search_dict, depth=d)
        
        # Prune best model
        if prune:
            search_dict = prune_best_model(search_dict, depth=d, lik=lik)
    
#     if d == max_depth:
#         print('Reached max depth, ending search.')
        
    # Look for best model
    best_model_name = min([(i[2], k) for k, i in search_dict.items()])[1]
    
    # Variance decomposition of best model
    var_percent = variance_contributions(
       search_dict[best_model_name][1], 
       best_model_name,
       lik=lik)
    
    # Return output
    return search_dict, edge_list, best_model_name, var_percent

def pred_kernel_parts(m, k_names, x_idx, lik='gaussian'):#, unit_idx, unit_label):
    """
    Breaks up kernel in model to plot separate pieces
    """
    
    # Get training data out of model
    X, Y = m.data
    
    # Get x min and max values
    x_min = min(X[:, x_idx])
    x_max = max(X[:, x_idx])
    
    # # Build prediction dataset
    # x_new = np.zeros_like(m.data[0])
    # x_new[:, x_idx] = np.linspace(x_min, x_max, m.data[0].shape[0])
    # x_new[:, unit_idx] = unit_label
    
    # Compute residuals
    mean_pred, var_pred = m.predict_y(m.data[0])
    resids = tf.cast(m.data[1], tf.float64) - mean_pred
    
    # Split kernel names by sum sign
    kernel_names = k_names.split('+')
    
    # Get variance pieces
    var_contribs = variance_contributions(m, k_names, lik)
    var_percent = [100*round(x/sum(var_contribs),2) for x in var_contribs]
    
    fig, ax = plt.subplots(nrows=len(kernel_names)+1,
                           sharex=True,
#                            sharey=True,
                           figsize=(10,7))
    c = 0
    if len(kernel_names) > 1: #m.kernel.name in ['sum']:
            for k in m.kernel.kernels:
#                 temp_m = gpflow.models.GPR(data=m.data,
#                                            kernel=k)
                # Specify model
                if lik == 'gaussian':
        #             m = gpflow.models.GPR(
                    temp_m = gpflow.models.VGP(
                        data=(X, Y),
                        kernel=k,
                        mean_function=m.mean_function,
                        likelihood=gpflow.likelihoods.Gaussian())
                elif lik == 'exponential':
                    temp_m = gpflow.models.VGP(
                        data=(X, Y),
                        kernel=k,
#                         mean_function=gpflow.mean_functions.Constant(),
                        likelihood=gpflow.likelihoods.Exponential())
                elif lik == 'poisson':
                    temp_m = gpflow.models.VGP(
                        data=(X, Y),
                        kernel=k,
#                         mean_function=gpflow.mean_functions.Constant(),
                        likelihood=gpflow.likelihoods.Poisson())
                elif lik == 'gamma':
                    temp_m = gpflow.models.VGP(
                        data=(X, Y),
                        kernel=k,
#                         mean_function=gpflow.mean_functions.Constant(),
                        likelihood=gpflow.likelihoods.Gamma())
                elif lik == 'bernoulli':
                    temp_m = gpflow.models.VGP(
                        data=(X, Y),
                        kernel=k,
                        mean_function=m.mean_function,
                        likelihood=gpflow.likelihoods.Bernoulli())
                else:
                    print('Unknown likelihood requested.')
                
                # Plot all possible category means if categorical
                if 'categorical' in kernel_names[c]:
                    for cat_idx in re.findall(r'categorical\[(\d+)\]', 
                                              kernel_names[c]):
                        cat_idx = int(cat_idx)
                        for cat_val in np.unique(X[:, cat_idx]):
                            x_new = np.zeros_like(X)
                            x_new[:, x_idx] = np.linspace(x_min, x_max, m.data[0].shape[0])
                            x_new[:, cat_idx] = cat_val
                            
                            mean, var = temp_m.predict_y(x_new)
                            ax[c].plot(x_new[:, x_idx],
                                       mean.numpy().flatten())
#                             ax[c].fill_between(
#                                 x_new[:, x_idx],
#                                 mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
#                                 mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
# #                                 color='lightgreen',
#                                 alpha=0.1,
#                             )
            
                else:
                    mean, var = temp_m.predict_y(x_new)
                    ax[c].plot(x_new[:, x_idx],
                               mean.numpy().flatten(),
                               color='darkgreen')
                    ax[c].fill_between(
                        x_new[:, x_idx],
                        mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                        color='lightgreen',
                        alpha=0.5,
                    )
                    
                    # Predict function samples
                    samples = temp_m.predict_f_samples(x_new, 10)
                    ax[c].plot(x_new[:,x_idx], 
                       samples[:, :, 0].numpy().T,# "C0", 
                       color='dimgray',
                       linewidth=0.5)
                
                # Add title for specific feature
                ax[c].set(title=f'{kernel_names[c]} ({var_percent[c]}%)')
                c+=1
    
    else:
        # This is if we only have one kernel component
        temp_m = m
                
        # Plot all possible category means if categorical
        if 'categorical' in kernel_names[c]:
            for cat_idx in re.findall(r'categorical\[(\d+)\]', 
                                      kernel_names[c]):
                cat_idx = int(cat_idx)
                for cat_val in np.unique(X[:, cat_idx]):
                    x_new = np.zeros_like(X)
                    x_new[:, x_idx] = np.linspace(x_min, x_max, m.data[0].shape[0])
                    x_new[:, cat_idx] = cat_val
                    
                    mean, var = temp_m.predict_y(x_new)
                    ax[c].plot(x_new[:, x_idx],
                               mean.numpy().flatten())
#                             ax[c].fill_between(
#                                 x_new[:, x_idx],
#                                 mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
#                                 mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
# #                                 color='lightgreen',
#                                 alpha=0.1,
#                             )
            
        else:
            mean, var = temp_m.predict_y(x_new)
            ax[c].plot(x_new[:, x_idx],
                       mean.numpy().flatten(),
                       color='darkgreen')
            ax[c].fill_between(
                x_new[:, x_idx],
                mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                color='lightgreen',
                alpha=0.5,
            )
        
        # Add title for specific feature
        ax[c].set(title=f'{kernel_names[c]} ({var_percent[c]}%)')
        c+=1
    
    # Plot residuals
    ax[c].plot(x_new[:, x_idx],
               np.zeros_like(x_new[:, x_idx]),
               color='darkgreen')
    error_sd = np.sqrt(m.parameters[-1].numpy())
    ax[c].fill_between(
        x_new[:, x_idx],
        -1.96 * error_sd,
        1.96 * error_sd,
        color='lightgreen',
        alpha=0.5
    )
    ax[c].scatter(m.data[0][:, x_idx],
                  resids,
                  color='black',
                  alpha=0.5)
    ax[c].set(title=f'residuals ({var_percent[c]}%)')
    
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
                prod_var *= k.variance.numpy().round(3)
            variance_list += [prod_var.tolist()]
        else:
            variance_list += [m.kernel.variance.numpy().round(3)]
    else:
        for k in range(len(kernel_names)):
            if m.kernel.kernels[k].name == 'product':
                prod_var = 1
                for k2 in m.kernel.kernels[k].kernels:
                    prod_var *= k2.variance.numpy().round(3)
                variance_list += [prod_var.tolist()]
            else:
                variance_list += [m.kernel.kernels[k].variance.numpy().round(3).tolist()]
        
    # Get likelihood variance
    if lik == 'gaussian':
        variance_list += [m.likelihood.variance.numpy().round(3).tolist()]
    else:
        variance_list += [np.std(calc_residuals(m))]
#     elif lik == 'exponential':
#     elif lik == 'poisson':
#     elif lik == 'gamma':
#     elif lik == 'bernoulli':
#         variance_list += 
#     else:
#         raise ValueError('Unknown likelihood function specified.')
    return variance_list
    
# def run_search_operator(X, Y, kern_list, cat_vars, max_depth, keep_all=False):
#     """
#     Run kernel search for each variable in Y independently. 

#     Parameters
#     ----------
#     X : TYPE
#         DESCRIPTION.
#     Y : TYPE
#         DESCRIPTION.
#     kern_list : TYPE
#         DESCRIPTION.
#     cat_vars : TYPE
#         DESCRIPTION.
#     max_depth : TYPE
#         DESCRIPTION.
#     keep_all : TYPE, optional
#         DESCRIPTION. The default is False.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.
#     edges : TYPE
#         DESCRIPTION.

#     """
    
#     # Dictionary to hold results for each variable
#     out_results = {}
#     out_figures = {}
#     num_y = 1 if len(Y.shape) == 1 else Y.shape[1]
#     for y_ in range(num_y):
#         print(f"Currently working on output {y_}")
#         # Flag for missing values 
#         x_idx = ~np.isnan(X).any(axis=1)
#         y_idx = ~np.isnan(Y[:, y_])
        
#         # Get complete cases        
#         comp_X = X[x_idx & y_idx]
#         comp_y = Y[x_idx & y_idx, y_]
        
#         res, edges = full_kernel_search(
#             X=comp_X.reshape(-1,X.shape[1]),
#             Y=comp_y.reshape(-1,1),
#             kern_list=kern_list,
#             cat_vars=cat_vars,
#             max_depth=max_depth,
#             keep_all=keep_all)
    
#         # Get best model
#         best_model = min([(i[2], k) for k, i in res.items()])
#         print(f'Best model for {y_}: {best_model[1]}')
#         print('')
    
#         # Plot marginal posterior
#         res_p = pred_kernel_parts(res[best_model[1]][1], 
#                       best_model[1],
#                       x_idx=2)
        
#         out_results[y_] = res
#         out_figures[y_] = res_p
#     return out_results, out_figures

def calc_residuals(m):
    """
    Calculate pearson residuals from model
    """

    # Get observed predictions and variance
    mean_y, var_y = m.predict_y(m.data[0])
    
    # Calculate standardized residuals
    resids = ((tf.cast(m.data[1], tf.float64) - mean_y)/np.sqrt(var_y)).numpy()
    
    return resids
    
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument
    
    Source: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution"""
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

###############################################################################
# Plotting functions
###############################################################################
