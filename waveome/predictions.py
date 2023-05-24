import numpy as np
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns
import re
f64 = gpflow.utilities.to_default_float
from .utilities import (
    calc_rsquare,
    calc_residuals, 
    print_kernel_names,
    replace_kernel_variables,
    individual_kernel_predictions
)

def pred_kernel_parts(
        m, 
        x_idx, 
        unit_idx, 
        col_names, 
        lik='gaussian', 
        x_idx_min = None, 
        x_idx_max = None
    ):
    """
    Breaks up kernel in model to plot separate pieces
    """
    
    # Copy model
    m_copy = gpflow.utilities.deepcopy(m)
    m_name = m_copy.name
    X = m_copy.data[0].numpy()
    Y = m_copy.data[1].numpy()

    # Set bounds on x-axis if not specified
    x_idx_min = X[:, x_idx].min() if x_idx_min == None else x_idx_min
    x_idx_max = X[:, x_idx].max() if x_idx_max == None else x_idx_max

    # Get variance pieces
    var_contribs = calc_rsquare(m=m_copy)
    var_percent = [100*round(x/sum(var_contribs),3) for x in var_contribs]
    
    # Get kernel names
    if m_copy.kernel.name == "constant":
        kernel_names = []
        fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
        plot_residuals(m, lik, x_idx, x_idx_min, x_idx_max, ax, 
            var_percent=100, col_names=col_names)
        return fig, ax
    else:
        kernel_list = print_kernel_names(m_copy.kernel, with_idx=True)
        # If it is a single kernel component then make list
        if type(kernel_list) != list:
            k_names = kernel_list
        else:
            k_names = '+'.join(kernel_list)

        # Split kernel names by sum sign
        kernel_names = k_names.split('+')
    
    # Make subplots for multiple components
    fig, ax = plt.subplots(ncols=len(kernel_names)+1,
                           # sharex=True,
                           sharey=True,
                           figsize=(5*(len(kernel_names)+1), 5))
    plot_idx = 0
    kernel_idx = 0
    for k_name in kernel_names: #m_copy.kernel.kernels:
        # print(f'k_name={k_name}')
        # Pull off specific kernel component
        if '*' in k_name and len(kernel_names) == 1:
            k = m_copy.kernel #.kernels[kernel_idx]
                #m_copy.kernel.kernels[kernel_idx] * m_copy.kernel.kernels[kernel_idx+1]
            kernel_idx += 1
        elif len(kernel_names) == 1:
            k = m_copy.kernel
        else:
            k = m_copy.kernel.kernels[kernel_idx]
            kernel_idx += 1
                
        # Plot all possible category means if categorical
        if 'categorical' in k_name: #kernel_names[c]:
            for cat_idx in re.findall(r'categorical\[(\d+)\]', 
                                      k_name): #kernel_names[c]):
                cat_idx = int(cat_idx)
                # Set up empyty dataset with domain support
                x_new = np.zeros((1000, m.data[0].shape[1]))
                x_new[:, x_idx] = np.linspace(
                        x_idx_min, 
                        x_idx_max, 
                        1000
                ) 
                
                # For each unique level of category replace and predict values
                num_unique_cat = len(np.unique(X[:, cat_idx]))
                for cat_val in np.unique(X[:, cat_idx]):                    
                    x_new[:, cat_idx] = cat_val
                    # mean, var = temp_m.predict_y(x_new)
                    mean, var, samps, cov = individual_kernel_predictions(
                        model=m_copy,
                        kernel_idx=np.argwhere([x == k_name for x in kernel_names])[0][0],
                        X=x_new,
                        white_noise_amt=1e-2)
                    mean = mean.numpy().flatten()
                    var = var.numpy().flatten()
                    
                    # Deal with transforming output if needed
                    if m_copy.likelihood.name != 'gaussian':
                        mean = m_copy.likelihood.invlink(mean).numpy().flatten()
                        upper_ci = m_copy.likelihood.invlink(mean + 1.96 * np.sqrt(var))
                        lower_ci = m_copy.likelihood.invlink(mean - 1.96 * np.sqrt(var))
                    else:
                        upper_ci = mean + 1.96 * np.sqrt(var)
                        lower_ci = mean - 1.96 * np.sqrt(var)

                    # Decide if we should annotate each category or not
                    if num_unique_cat < 5:
                        ax[plot_idx].plot(
                            x_new[:, x_idx],
                            mean,
                            alpha=0.5,
                            label=cat_val
                        )
                        ax[plot_idx].fill_between(
                            x_new[:, x_idx],
                            lower_ci, # mean - 1.96 * np.sqrt(var),
                            upper_ci, # mean + 1.96 * np.sqrt(var),
                            color='lightgreen',
                            alpha=0.5,
                        )
                        
                        # If last category then add legend to plot
                        if cat_val == num_unique_cat - 1:
                            ax[plot_idx].legend(loc="upper right")

                    else:
                        ax[plot_idx].plot(
                            x_new[:, x_idx],
                            mean,
                            alpha=0.5
                        )
                        
                # Set the subplot title to match the true variable name
                ax[plot_idx].set(
                    xlabel=f"{replace_kernel_variables('['+str(x_idx)+']', col_names).strip('[]')}"
                )

        # Deal with interaction if two continuous features
        # but only if the two continuous features aren't the same 
        elif '*' in k_name and len(np.unique(re.findall(r'\[(\d+)\]', k_name))) > 1:
            # Grab all of the variable indexes
            x_idxs = [int(x) for x in re.findall(r'\[(\d+)\]', k_name)]
            x_new = np.zeros((1000, m.data[0].shape[1]))
            # Choose the first one as the main support
            x_new[:, x_idxs[0]] = np.linspace(
                X[:, x_idxs[0]].min(), 
                X[:, x_idxs[0]].max(), 
                1000
            )         
            # Get quantiles of the others (five number summary)
            for i in np.percentile(X[:, x_idxs[1]], q=[0, 25, 50, 75, 100]):
                x_new[:, x_idxs[1]] = i
                # mean, var = temp_m.predict_y(x_new)
                mean, var, samps, cov = individual_kernel_predictions(
                    model=m_copy,
                    kernel_idx=np.argwhere([x == k_name for x in kernel_names])[0][0],
                    X=x_new)
                mean = mean.numpy().flatten()
                var = var.numpy().flatten()
                
                # Deal with transforming output if needed
                if m_copy.likelihood.name != 'gaussian':
                    mean = m_copy.likelihood.invlink(mean).numpy().flatten()
                    upper_ci = m_copy.likelihood.invlink(mean + 1.96 * np.sqrt(var))
                    lower_ci = m_copy.likelihood.invlink(mean - 1.96 * np.sqrt(var))
                else:
                    upper_ci = mean + 1.96 * np.sqrt(var)
                    lower_ci = mean - 1.96 * np.sqrt(var)
                
                ax[plot_idx].plot(
                    x_new[:, x_idxs[0]],
                    mean,
                    alpha=0.5,
                    label=round(i, 1)
                )
                ax[plot_idx].fill_between(
                    x_new[:, x_idxs[0]],
                    lower_ci, # mean - 1.96 * np.sqrt(var),
                    upper_ci, # mean + 1.96 * np.sqrt(var),
                    color='lightgreen',
                    alpha=0.5,
                )
            ax[plot_idx].legend()
            ax[plot_idx].set(
                xlabel=f"{replace_kernel_variables('['+str(x_idxs[0])+']', col_names).strip('[]')}"
            )

        # Otherwise standard decomposition
        else:
            # Grab variable index
            if 'constant' in k_name:
                var_idx = x_idx
            else:
                var_idx = int(re.findall(r'\[(\d+)\]', k_name)[0]) #kernel_names[c])[0])

            # Add full range of x axis
            x_new = np.zeros((1000, m.data[0].shape[1]))
            x_new[:, var_idx] = np.linspace(
                X[:, var_idx].min(),
                X[:, var_idx].max(),
                1000
            )
            
            # Pull predictions from specific component
            mean, var, samps, cov = individual_kernel_predictions(
                model=m_copy,
                kernel_idx=np.argwhere([x == k_name for x in kernel_names])[0][0],
                X=x_new)
            mean = mean.numpy().flatten()
            var = var.numpy().flatten()
            
            # Deal with transforming output if needed
            if m_copy.likelihood.name != 'gaussian':
                mean = m_copy.likelihood.invlink(mean).numpy().flatten()
                upper_ci = m_copy.likelihood.invlink(mean + 1.96 * np.sqrt(var))
                lower_ci = m_copy.likelihood.invlink(mean - 1.96 * np.sqrt(var))
                samps = m_copy.likelihood.invlink(samps)
            else:
                upper_ci = mean + 1.96 * np.sqrt(var)
                lower_ci = mean - 1.96 * np.sqrt(var)
            
            p = sns.lineplot(x=x_new[:, var_idx],
#                  y=mean.numpy().flatten(),
             y=mean,
             linewidth=2.5,
                color='darkgreen',
                ax=ax[plot_idx])

            ax[plot_idx].fill_between(
                x_new[:, var_idx],
                lower_ci, # mean - 1.96 * np.sqrt(var),
                upper_ci, # mean + 1.96 * np.sqrt(var),
                color='lightgreen',
                alpha=0.5,
            )
            ax[plot_idx].plot(x_new[:, var_idx], 
                   samps,#[:, :, 0].numpy().T,# "C0", 
                   color='dimgray',
                   linewidth=0.5,
                   alpha=0.2)
        #     plt.close()

            ax[plot_idx].set(
                xlabel=f"{replace_kernel_variables('['+str(var_idx)+']', col_names).strip('[]')}",
                #title=f"{replace_kernel_variables(str(x_idx), col_names)}"
            )
            # gp_predict_fun(
            #     gp=temp_m,
            #     x_idx=var_idx,
            #     unit_idx=unit_idx,
            #     ax=ax[plot_idx],
            #     plot_points=False,
            #     col_names=col_names
            # )
            

        # Add title for specific feature
        ax[plot_idx].set(title=f'{replace_kernel_variables(k_name, col_names)} ({round(var_percent[plot_idx], 1)}%)')
        plot_idx+=1
    
    # Plot residuals
    plot_residuals(m, lik, x_idx, x_idx_min, x_idx_max, ax[plot_idx], 
                   var_percent=var_percent[plot_idx], col_names=col_names)
    
    # Adjust scale if needed
    if lik == 'gamma':
        for ax_ in ax:
            ax_.set_yscale('log')
    
    return fig, ax

def plot_residuals(m, lik, x_idx, x_idx_min, x_idx_max, ax, var_percent, col_names):

    # Compute residuals
    # mean_pred, var_pred = m.predict_y(m.data[0])
    resids = calc_residuals(m) #tf.cast(m.data[1], tf.float64) - mean_pred

    if lik == 'gaussian':
        x_resid = np.linspace(
            x_idx_min, # X[:, x_idx].min(), 
            x_idx_max, # X[:, x_idx].max(), 
            1000
        )
        ax.plot(
            x_resid, #x_new[:, x_idx],
            np.zeros(len(x_resid)), #np.zeros_like(x_new[:, x_idx]),
            color='darkgreen',
            linewidth=2.5
        )
        # error_sd = np.sqrt(m.parameters[-1].numpy())
        # Calculate the model residuals to get the standard deviation
        error_sd = np.std(resids)
        ax.fill_between(
            x_resid, #x_new[:, x_idx],
            -1.96 * error_sd,
            1.96 * error_sd,
            color='lightgreen',
            alpha=0.5
        )
        ax.scatter(m.data[0][:, x_idx],
                      resids,
                      color='black',
                      alpha=0.5,
                      s=20)
    else:
        x_resid = np.linspace(
            x_idx_min, # X[:, x_idx].min(), 
            x_idx_max, # X[:, x_idx].max(), 
            1000
        )
        ax.plot(
            x_resid, #x_new[:, x_idx],
            m.likelihood.invlink(np.zeros(len(x_resid))),
            color='darkgreen',
            linewidth=2.5
        )
        # error_sd = np.sqrt(m.parameters[-1].numpy())
        # Calculate the model residuals to get the standard deviation
        error_sd = np.std(resids)
        ax.fill_between(
            x_resid, #x_new[:, x_idx],
            m.likelihood.invlink(-1.96 * error_sd),
            m.likelihood.invlink(1.96 * error_sd),
            color='lightgreen',
            alpha=0.5
        )
        ax.scatter(m.data[0][:, x_idx],
                      m.likelihood.invlink(resids),
                      color='black',
                      alpha=0.5,
                      s=20)
    ax.set(title=f'residuals ({round(var_percent, 1)}%)',
            xlabel=f"{replace_kernel_variables('['+str(x_idx)+']', col_names).strip('[]')}")

def gp_predict_fun(gp,  # X, Y, x_min, x_max,
                   x_idx, unit_idx, col_names,
                   unit_label=None, num_funs=10,
                   ax=None, plot_points=True):
    """
    Plot marginal closed-form posterior distribution.
    """

    # Pull training data from model
    X_train, Y_train = gp.data
    X_train = X_train.numpy()
    Y_train = Y_train.numpy()

    # Create test points
# #     x_new = np.zeros_like(X)
# #     x_new[:,x_idx] = np.linspace(x_min, x_max, X.shape[0])
# #     x_new[:, unit_idx] = unit_label
#     x_new = np.zeros((1000, X_train.shape[1]))
#     x_new[:, x_idx] = np.linspace(X_train[:, x_idx].min(),
#                                   X_train[:, x_idx].max(),
#                                   1000)
#     x_new[:, unit_idx] = unit_label
    
    if unit_idx != None:
        x_new = np.tile(
            A=np.median(
                X_train[X_train[:, unit_idx] == unit_label, ],
                axis=0
            ),
            reps=(1000, 1)
        )
    else:
        x_new = np.tile(
            A=np.median(
                X_train,
                axis=0
            ),
            reps=(1000, 1)
        )
    # Add full range of x axis
    x_new[:, x_idx] = np.linspace(
        X_train[:, x_idx].min(),
        X_train[:, x_idx].max(),
        1000
    )
    
    # Predict mean and variance on new data
#     mean, var = gp.predict_f(x_new)
    mean, var = gp.predict_y(x_new)
    # print('observed:', mean.numpy().flatten()[:5], var.numpy().flatten()[:5])
    mean_f, var_f = gp.predict_f(x_new)
    # print('latent function:', mean_f.numpy().flatten()[:5], var_f.numpy().flatten()[:5])
    
#     return(mean.numpy()[:5], mean_f.numpy()[:5])
    
    # Pull some posterior functions
    tf.random.set_seed(1) 
    samples = gp.predict_f_samples(x_new, num_funs)
    samples = samples[:, :, 0].numpy().T 

    # Transform function samples if not Gaussian
    if gp.likelihood.name == 'gamma':
        samples = gp.likelihood.shape*gp.likelihood.invlink(samples)#.numpy() 
        upper_ci = gp.likelihood.shape*gp.likelihood.invlink(mean_f + 1.96*np.sqrt(var_f)).numpy().flatten()
        lower_ci = gp.likelihood.shape*gp.likelihood.invlink(mean_f - 1.96*np.sqrt(var_f)).numpy().flatten()
    elif gp.likelihood.name == 'bernoulli':
        samples = gp.likelihood.invlink(samples)
        upper_ci = gp.likelihood.invlink(mean_f + 1.96*np.sqrt(var_f)).numpy().flatten()
        # print(upper_ci[:5])
        lower_ci = gp.likelihood.invlink(mean_f - 1.96*np.sqrt(var_f)).numpy().flatten()
        # print('lower:', lower_ci[:5])
    else:
        lower_ci = mean_f[:, 0] - 1.96 * np.sqrt(var_f[:, 0])
        upper_ci = mean_f[:, 0] + 1.96 * np.sqrt(var_f[:, 0])
    # Generate plot
#     p = plt.figure(figsize=(10, 5))
    if ax == None:
        fig, ax = plt.subplots(figsize=(10,5))
#     p = sns.scatterplot(x=X[:,x_idx],
#                     y=Y.flatten(),
#                     hue=X[:,unit_idx].astype(int).astype(str),
#                         legend=False)
    # Do we want to plot individual points?
    if plot_points:
        # Do we want these points to be weighted by a single unit?
        if unit_idx != None:
            person_rows = X_train[:,unit_idx] == unit_label
            p = sns.scatterplot(x=X_train[person_rows,x_idx],
                            y=Y_train.flatten()[person_rows],
        #                     hue=X[:,unit_idx].astype(int).astype(str),
        #                         legend=False
                               s=100,
                               color='black',
                               ax=ax)
            p = sns.scatterplot(x=X_train[~np.array(person_rows),x_idx],
                            y=Y_train.flatten()[~np.array(person_rows)],
        #                     hue=X[:,unit_idx].astype(int).astype(str),
        #                         legend=False
                               s=20,
                               color='grey',
                               ax=ax)
        else:
            p = sns.scatterplot(x=X_train[: ,x_idx],
                                y=Y_train.flatten(),
                                s=20,
                                color='grey',
                                ax=ax)
    
    p = sns.lineplot(x=x_new[:,x_idx],
#                  y=mean.numpy().flatten(),
                 y=mean.numpy().flatten(),
                 linewidth=2.5,
                    color='darkgreen',
                    ax=ax)

    ax.fill_between(
        x_new[:, x_idx],
        upper_ci, #mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
        lower_ci, #mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color='lightgreen',
        alpha=0.5,
    )
    ax.plot(x_new[:,x_idx], 
           samples,#[:, :, 0].numpy().T,# "C0", 
           color='dimgray',
           linewidth=0.5,
           alpha=0.2)
#     plt.close()

    ax.set(
        xlabel=f"{replace_kernel_variables('['+str(x_idx)+']', col_names).strip('[]')}",
        #title=f"{replace_kernel_variables(str(x_idx), col_names)}"
    )
    return(ax)