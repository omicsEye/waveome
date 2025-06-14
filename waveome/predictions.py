import copy
import re

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from .utilities import (
    calc_deviance_explained_components,
    calc_residuals,
    calc_rsquare,
    individual_kernel_predictions,
    print_kernel_names,
    replace_kernel_variables,
)

# import tensorflow_probability as tfp


f64 = gpflow.utilities.to_default_float


def pred_kernel_parts(
    m,
    x_idx,
    col_names,
    data=None,
    var_explained=None,
    categorical_dict={},
    lik="gaussian",
    marginal=True,
    x_idx_min=None,
    x_idx_max=None,
    num_cols_in_fig=4,
    figsize=None,
    sharey=False,
    conf_level_val=1.96,
):
    """
    Breaks up kernel in model to plot separate pieces
    """

    # Copy model
    m_copy = gpflow.utilities.deepcopy(m)

    if data is not None:
        X = data[0]
        Y = data[1]
    elif isinstance(m_copy.data[0], np.ndarray):
        X = m_copy.data[0]
        Y = m_copy.data[1]
    else:
        X = m_copy.data[0].numpy()
        Y = m_copy.data[1].numpy()

    # Set bounds on x-axis if not specified
    x_idx_min = X[:, x_idx].min() if x_idx_min is None else x_idx_min
    x_idx_max = X[:, x_idx].max() if x_idx_max is None else x_idx_max

    # Get variance pieces
    # var_contribs = calc_rsquare(m=m_copy)
    if var_explained is None:
        var_contribs = calc_deviance_explained_components(model=m_copy, data=data)
    else:
        var_contribs = copy.deepcopy(var_explained)
    # var_percent = [100 * round(x / sum(var_contribs), 3) for x in var_contribs]
    # var_percent = [100 * x for x in var_contribs]
    var_percent = var_contribs
    var_percent[-1] *= 100

    # Get kernel names
    # TODO: Fix this issue up, empty kernel does not produce the correct
    # plots and variance percent should not be 100% if there is a mean
    # maybe?
    if m_copy.kernel.name in ["constant", "empty"]:
        kernel_names = []
        fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
        plot_residuals(
            m,
            data,
            lik,
            x_idx,
            x_idx_min,
            x_idx_max,
            ax,
            var_percent=100,
            col_names=col_names,
            conf_level_val=conf_level_val,
        )
        return fig, ax
    else:
        kernel_list = print_kernel_names(m_copy.kernel, with_idx=True)
        # If it is a single kernel component then make list
        if isinstance(kernel_list, list) is False:
            k_names = kernel_list
        else:
            k_names = "+".join(kernel_list)

        # Split kernel names by sum sign
        kernel_names = k_names.split("+")

    # Make subplots for multiple components
    num_figs = len(kernel_names) + 1
    num_rows = int(np.ceil(num_figs / num_cols_in_fig))
    if figsize is None:
        figsize = (7.2, 1.44 * num_rows)#(5 * num_cols_in_fig, 5 * num_rows)
    fig, ax = plt.subplots(
        ncols=num_cols_in_fig,  # len(kernel_names)+1,
        nrows=num_rows,
        # sharex=True,
        sharey=sharey,
        figsize=figsize,
        squeeze=False,
    )

    plot_row_idx = 0
    plot_col_idx = 0
    kernel_idx = 0
    for plot_idx, k_name in enumerate(kernel_names):  # m_copy.kernel.kernels:
        # print(f'k_name={k_name}')
        # Pull off specific kernel component
        if "*" in k_name and len(kernel_names) == 1:
            # k = m_copy.kernel
            kernel_idx += 1
            product_term = True
        # elif len(kernel_names) == 1:
        # k = m_copy.kernel
        else:
            # k = m_copy.kernel.kernels[kernel_idx]
            kernel_idx += 1
            product_term = False

        # Plot all possible category means if categorical
        if "categorical" in k_name:  # kernel_names[c]:

            # Check to see if categorical is crossed with a continuous feature
            if "*" in k_name:
                # Grab all of the variable indexes
                x_idxs = [int(x) for x in re.findall(r"\[(\d+)\]", k_name)]
                x_new = np.zeros((1000, X.shape[1]))
                # Choose the second one as the main support
                x_new[:, x_idxs[1]] = np.linspace(
                    X[:, x_idxs[1]].min(), X[:, x_idxs[1]].max(), 1000
                )
                plot_x_idx = x_idxs[1]
            else:
                # Set up empyty dataset with domain support
                x_new = np.zeros((1000, X.shape[1]))
                x_new[:, x_idx] = np.linspace(x_idx_min, x_idx_max, 1000)
                plot_x_idx = x_idx

            for cat_idx in re.findall(
                r"categorical\[(\d+)\]", k_name
            ):  # kernel_names[c]):
                cat_idx = int(cat_idx)
                # # Set up empyty dataset with domain support
                # x_new = np.zeros((1000, X.shape[1]))
                # x_new[:, x_idx] = np.linspace(x_idx_min, x_idx_max, 1000)

                # For each unique level of category replace and predict values
                num_unique_cat = len(np.unique(X[:, cat_idx]))
                for cat_val in np.unique(X[:, cat_idx]):
                    x_new[:, cat_idx] = cat_val
                    # mean, var = temp_m.predict_y(x_new)
                    mean, var, samps, cov = individual_kernel_predictions(
                        model=m_copy,
                        kernel_idx=np.argwhere(
                            [x == k_name for x in kernel_names]
                        )[0][0],
                        data=data,
                        product_term=product_term,
                        X=x_new,
                        marginal=marginal,
                        white_noise_amt=1e-2,
                    )
                    mean = mean.numpy().flatten()
                    var = var.numpy().flatten()

                    # Transform output
                    mean_resp = m_copy.likelihood._conditional_mean(
                        X=(
                            x_new
                            if m_copy.likelihood.name != "gaussian"
                            else x_new[:, 0]
                        ),
                        F=mean,
                    )
                    upper_ci = m_copy.likelihood._conditional_mean(
                        X=(
                            x_new
                            if m_copy.likelihood.name != "gaussian"
                            else x_new[:, 0]
                        ),
                        F=mean + conf_level_val * np.sqrt(var),
                    )
                    lower_ci = m_copy.likelihood._conditional_mean(
                        X=(
                            x_new
                            if m_copy.likelihood.name != "gaussian"
                            else x_new[:, 0]
                        ),
                        F=mean - conf_level_val * np.sqrt(var),
                    )

                    # Decide if we should annotate each category or not
                    if num_unique_cat <= 5:
                        ax[plot_row_idx, plot_col_idx].plot(
                            x_new[:, plot_x_idx],
                            mean_resp,
                            alpha=0.5,
                            label=(
                                cat_val
                                if col_names[cat_idx]
                                not in categorical_dict.keys()
                                else categorical_dict[col_names[cat_idx]][1][
                                    int(cat_val)
                                ]
                            ),
                        )
                        ax[plot_row_idx, plot_col_idx].fill_between(
                            x_new[:, plot_x_idx],
                            lower_ci,  # mean - conf_level_val * np.sqrt(var),
                            upper_ci,  # mean + conf_level_val * np.sqrt(var),
                            color="lightgreen",
                            alpha=0.5,
                        )

                        # If last category then add legend to plot
                        if cat_val == num_unique_cat - 1:
                            ax[plot_row_idx, plot_col_idx].legend(
                                loc="upper right"
                            )

                    else:
                        ax[plot_row_idx, plot_col_idx].plot(
                            x_new[:, plot_x_idx], mean_resp, alpha=0.5
                        )

                # Set the subplot title to match the true variable name
                ax[plot_row_idx, plot_col_idx].set(
                    xlabel=(
                        f"""{replace_kernel_variables(
                            '['+str(plot_x_idx)+']',
                            col_names
                        ).strip('[]')}"""
                    )
                )

        # Deal with interaction if two continuous features
        # but only if the two continuous features aren't the same
        elif (
            "*" in k_name
            and len(np.unique(re.findall(r"\[(\d+)\]", k_name))) > 1
        ):
            # Grab all of the variable indexes
            x_idxs = [int(x) for x in re.findall(r"\[(\d+)\]", k_name)]
            x_new = np.zeros((1000, X.shape[1]))
            # Choose the first one as the main support
            x_new[:, x_idxs[0]] = np.linspace(
                X[:, x_idxs[0]].min(), X[:, x_idxs[0]].max(), 1000
            )
            # Get quantiles of the others (five number summary)
            for i in np.percentile(X[:, x_idxs[1]], q=[0, 25, 50, 75, 100]):
                x_new[:, x_idxs[1]] = i
                # mean, var = temp_m.predict_y(x_new)
                mean, var, samps, cov = individual_kernel_predictions(
                    model=m_copy,
                    kernel_idx=np.argwhere(
                        [x == k_name for x in kernel_names]
                    )[0][0],
                    data=data,
                    product_term=product_term,
                    X=x_new,
                    marginal=marginal
                )
                mean = mean.numpy().flatten()
                var = var.numpy().flatten()

                # Transform output
                mean_resp = m_copy.likelihood._conditional_mean(
                    X=(
                        x_new
                        if m_copy.likelihood.name != "gaussian"
                        else x_new[:, 0]
                    ),
                    F=mean,
                )
                upper_ci = m_copy.likelihood._conditional_mean(
                    X=(
                        x_new
                        if m_copy.likelihood.name != "gaussian"
                        else x_new[:, 0]
                    ),
                    F=mean + conf_level_val * np.sqrt(var),
                )
                lower_ci = m_copy.likelihood._conditional_mean(
                    X=(
                        x_new
                        if m_copy.likelihood.name != "gaussian"
                        else x_new[:, 0]
                    ),
                    F=mean - conf_level_val * np.sqrt(var),
                )

                # Decide if we should annotate each category or not

                ax[plot_row_idx, plot_col_idx].plot(
                    x_new[:, x_idxs[0]],
                    mean_resp,
                    alpha=0.5,
                    label=round(i, 1),
                )
                ax[plot_row_idx, plot_col_idx].fill_between(
                    x_new[:, x_idxs[0]],
                    lower_ci,  # mean - conf_level_val * np.sqrt(var),
                    upper_ci,  # mean + conf_level_val * np.sqrt(var),
                    color="lightgreen",
                    alpha=0.5,
                )
            ax[plot_row_idx, plot_col_idx].legend()
            ax[plot_row_idx, plot_col_idx].set(
                xlabel=(
                    f"""{replace_kernel_variables(
                        '['+str(x_idxs[0])+']',
                          col_names
                    ).strip('[]')}"""
                )
            )

        # Otherwise standard decomposition
        else:
            # Grab variable index
            if "constant" in k_name:
                var_idx = x_idx
            else:
                var_idx = int(
                    re.findall(r"\[(\d+)\]", k_name)[0]
                )  # kernel_names[c])[0])

            # Add full range of x axis
            x_new = np.zeros((1000, X.shape[1]))
            x_new[:, var_idx] = np.linspace(
                X[:, var_idx].min(), X[:, var_idx].max(), 1000
            )

            # Pull predictions from specific component
            mean, var, samps, cov = individual_kernel_predictions(
                model=m_copy,
                kernel_idx=np.argwhere([x == k_name for x in kernel_names])[0][
                    0
                ],
                data=data,
                product_term=product_term,
                X=x_new,
                marginal=marginal
            )
            mean = mean.numpy().flatten()
            var = var.numpy().flatten()

            # Transform output
            mean_resp = m_copy.likelihood._conditional_mean(
                X=(
                    x_new
                    if m_copy.likelihood.name != "gaussian"
                    else x_new[:, 0]
                ),
                F=mean,
            )
            upper_ci = m_copy.likelihood._conditional_mean(
                X=(
                    x_new
                    if m_copy.likelihood.name != "gaussian"
                    else x_new[:, 0]
                ),
                F=mean + conf_level_val * np.sqrt(var),
            )
            lower_ci = m_copy.likelihood._conditional_mean(
                X=(
                    x_new
                    if m_copy.likelihood.name != "gaussian"
                    else x_new[:, 0]
                ),
                F=mean - conf_level_val * np.sqrt(var),
            )
            samps_resp = m_copy.likelihood._conditional_mean(
                X=(
                    x_new
                    if m_copy.likelihood.name != "gaussian"
                    else x_new[:, 0]
                ),
                F=samps,
            )

            _ = sns.lineplot(
                x=x_new[:, var_idx],
                #                  y=mean.numpy().flatten(),
                y=mean_resp,
                linewidth=2.5,
                color="darkgreen",
                ax=ax[plot_row_idx, plot_col_idx],
            )

            ax[plot_row_idx, plot_col_idx].fill_between(
                x_new[:, var_idx],
                lower_ci,  # mean - conf_level_val * np.sqrt(var),
                upper_ci,  # mean + conf_level_val * np.sqrt(var),
                color="lightgreen",
                alpha=0.5,
            )
            ax[plot_row_idx, plot_col_idx].plot(
                x_new[:, var_idx],
                samps_resp,  # [:, :, 0].numpy().T,# "C0",
                color="dimgray",
                linewidth=0.5,
                alpha=0.2,
            )
            #     plt.close()

            ax[plot_row_idx, plot_col_idx].set(
                xlabel=(
                    f"""{replace_kernel_variables(
                        '['+str(var_idx)+']',
                        col_names
                    ).strip('[]')}"""
                )
            )

        # Add title for specific feature
        # split product terms over two lines
        ax[plot_row_idx, plot_col_idx].set(
            title=(
                f"""{replace_kernel_variables(
                    k_name,
                    col_names
                ).replace('*', '*'+chr(10))}"""
                f"""({round(var_percent[plot_idx], 1)})"""
            )
        )
        # Reset col index if at end and increment row index
        if plot_col_idx == (num_cols_in_fig - 1):
            plot_col_idx = 0
            plot_row_idx += 1
        else:
            plot_col_idx += 1

        plot_idx += 1

    # Plot residuals
    plot_residuals(
        m,
        data,
        lik,
        x_idx,
        x_idx_min,
        x_idx_max,
        ax[plot_row_idx, plot_col_idx],
        var_percent=var_percent[plot_idx],
        col_names=col_names,
        conf_level_val=conf_level_val,
        resid_type="pearson"
    )

    # Remove empty plots in last row
    for i in range(plot_col_idx + 1, num_cols_in_fig):
        fig.delaxes(ax[plot_row_idx, i])

    # Adjust scale if needed
    if lik == "gamma":
        for ax_ in ax:
            ax_.set_yscale("log")

    # Condense down plot space to make more tidy
    plt.tight_layout()

    return fig, ax


def plot_residuals(
    m,
    data,
    lik,
    x_idx,
    x_idx_min,
    x_idx_max,
    ax,
    var_percent,
    col_names,
    conf_level_val=1.96,
    resid_type="raw"
):
    # Compute residuals
    mean_pred, var_pred = m.predict_y(data[0])
    resids = calc_residuals(m, X=data[0], Y=data[1], resid_type=resid_type)
    ax.scatter(
        mean_pred,
        resids,
        color="black",
        alpha=0.5,
        s=20
    )
    # ax.scatter(data[0][:, x_idx], resids, color="black", alpha=0.5, s=20)
    # TODO: Think about line of best fit here maybe?

    # if lik == 'gaussian':
    #     x_resid = np.linspace(
    #         x_idx_min, # X[:, x_idx].min(),
    #         x_idx_max, # X[:, x_idx].max(),
    #         1000
    #     )
    #     ax.plot(
    #         x_resid, #x_new[:, x_idx],
    #         np.zeros(len(x_resid)), #np.zeros_like(x_new[:, x_idx]),
    #         color='darkgreen',
    #         linewidth=2.5
    #     )
    #     # error_sd = np.sqrt(m.parameters[-1].numpy())
    #     # Calculate the model residuals to get the standard deviation
    #     error_sd = np.std(resids)
    #     ax.fill_between(
    #         x_resid, #x_new[:, x_idx],
    #         -conf_level_val * error_sd,
    #         conf_level_val * error_sd,
    #         color='lightgreen',
    #         alpha=0.5
    #     )
    #     ax.scatter(m.data[0][:, x_idx],
    #                   resids,
    #                   color='black',
    #                   alpha=0.5,
    #                   s=20)
    # else:
    #     x_resid = np.linspace(
    #         x_idx_min, # X[:, x_idx].min(),
    #         x_idx_max, # X[:, x_idx].max(),
    #         1000
    #     )
    #     ax.plot(
    #         x_resid, #x_new[:, x_idx],
    #         m.likelihood.invlink(np.zeros(len(x_resid))),
    #         color='darkgreen',
    #         linewidth=2.5
    #     )
    #     # error_sd = np.sqrt(m.parameters[-1].numpy())
    #     # Calculate the model residuals to get the standard deviation
    #     error_sd = np.std(resids)
    #     ax.fill_between(
    #         x_resid, #x_new[:, x_idx],
    #         m.likelihood.invlink(-conf_level_val * error_sd),
    #         m.likelihood.invlink(conf_level_val * error_sd),
    #         color='lightgreen',
    #         alpha=0.5
    #     )
    #     ax.scatter(m.data[0][:, x_idx],
    #                   m.likelihood.invlink(resids),
    #                   color='black',
    #                   alpha=0.5,
    #                   s=20)
    ax.set(
        title=f"residuals ({round(var_percent, 1)}%)",
        # xlabel=(
        #     f"""{replace_kernel_variables(
        #         '['+str(x_idx)+']',
        #         col_names
        #     ).strip('[]')}"""
        # ),
        xlabel="fitted value",
        ylabel=f"{resid_type} residual"
    )


def gp_predict_fun(
    m,
    x_idx,
    col_names,
    X=None,
    Y=None,
    x_min=None,
    x_max=None,
    unit_idx=None,
    unit_label=None,
    num_funs=100,
    ref_quantile=0.5,
    return_vals=False,
    predict_type="mean",
    # predict_y=False,
    conf_level_val=1.96,
    label=None,
    cat_color_pal=sns.color_palette("Set1"),
    ax=None,
    plot_points=True,
):
    """
    Plot marginal closed-form posterior distribution.
    """

    # Pull training data from model if needed
    if X is None and Y is None:
        X_train, Y_train = m.data
        X_train = X_train.numpy()
        Y_train = Y_train.numpy()
    else:
        X_train = X
        Y_train = Y

    # Create test points
    # If no reference point given then use median of other values
    if unit_idx is not None and unit_label is not None:
        # assert unit_label is not None, print("Need unit_label with unit_idx")
        x_new = np.tile(
            A=np.quantile(
                X_train[X_train[:, unit_idx] == unit_label,],
                axis=0,
                q=ref_quantile,
            ),
            reps=(1000, 1),
        )
    elif unit_idx is not None and unit_label is None:
        # Predict for a "new" person using median attributes
        x_new = np.tile(
            A=np.quantile(X_train, axis=0, q=ref_quantile), reps=(1000, 1)
        )
        x_new[:, unit_idx] == np.inf
    else:
        x_new = np.tile(
            A=np.quantile(X_train, axis=0, q=ref_quantile), reps=(1000, 1)
        )
    # Add full range of x axis
    if x_min is None:
        x_min = X_train[:, x_idx].min()
    if x_max is None:
        x_max = X_train[:, x_idx].max()

    x_new[:, x_idx] = np.linspace(x_min, x_max, 1000)

    # Predict mean and variance on new data
    # Make sure we predict observed/latent function values
    # if predict_y is True:
    #     mean, var = m.predict_y(x_new)
    # else:
    #     mean, var = m.predict_f(x_new)
    mean, var = m.predict_f(x_new)

    # Pull some posterior functions
    tf.random.set_seed(1)
    samples = m.predict_f_samples(x_new, num_funs)
    samples = samples[:, :, 0].numpy().T

    # Return prediction values if that is all that we want
    if return_vals is True:
        return x_new, mean, var, samples

    # Transform function samples if not Gaussian
    # predict_type = 'obs' is complicated outside of gaussian likelihood
    # would need to have access to CDF to get likelihood intervals
    assert predict_type in [
        "mean",
        "obs",
        "func",
    ], "Unclear prediction type. ['mean', 'obs', 'func'] allowed."
    if predict_type == "mean":
        orig_mean = mean.numpy().copy()
        mean = m.likelihood.conditional_mean(x_new, mean)
        samples = m.likelihood.conditional_mean(x_new, samples)
        upper_ci = (
            m.likelihood.conditional_mean(
                x_new, orig_mean + conf_level_val * np.sqrt(var)
            )
            .numpy()
            .flatten()
        )
        lower_ci = (
            m.likelihood.conditional_mean(
                x_new, orig_mean - conf_level_val * np.sqrt(var)
            )
            .numpy()
            .flatten()
        )

    elif predict_type == "obs":
        assert m.likelihood.name == "gaussian", (
            "predict_type == 'obs' currently only works"
            " for 'gaussian' likelihoods"
        )
        mean, var = m.predict_y(x_new)
        samples = m.likelihood.conditional_mean(x_new, samples)
        lower_ci = (mean - conf_level_val * np.sqrt(var)).numpy().flatten()
        upper_ci = (mean + conf_level_val * np.sqrt(var)).numpy().flatten()
    else:  # predict_type == 'func':
        lower_ci = (mean - conf_level_val * np.sqrt(var)).numpy().flatten()
        upper_ci = (mean + conf_level_val * np.sqrt(var)).numpy().flatten()

    # Generate plot
    #     p = plt.figure(figsize=(10, 5))
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
    #     p = sns.scatterplot(x=X[:,x_idx],
    #                     y=Y.flatten(),
    #                     hue=X[:,unit_idx].astype(int).astype(str),
    #                         legend=False)
    # Do we want to plot individual points?
    if plot_points:
        # Do we want these points to be weighted by a single unit?
        if unit_idx is not None:
            person_rows = X_train[:, unit_idx] == unit_label
            _ = sns.scatterplot(
                x=X_train[person_rows, x_idx],
                y=Y_train.flatten()[person_rows],
                #                     hue=X[:,unit_idx].astype(int).astype(str),
                #                         legend=False
                s=100,
                color="black",
                linewidths=0.5,
                ax=ax,
            )
            _ = sns.scatterplot(
                x=X_train[~np.array(person_rows), x_idx],
                y=Y_train.flatten()[~np.array(person_rows)],
                #                     hue=X[:,unit_idx].astype(int).astype(str),
                #                         legend=False
                s=20,
                color="grey",
                ax=ax,
            )
        else:
            _ = sns.scatterplot(
                x=X_train[:, x_idx],
                y=Y_train.flatten(),
                s=20,
                color="grey",
                ax=ax,
            )

    # Do we want to use the default green color?
    if label is None:
        _ = sns.lineplot(
            x=x_new[:, x_idx],
            y=mean.numpy().flatten(),
            linewidth=2.5,
            color="darkgreen",
            ax=ax,
        )
        ax.fill_between(
            x_new[:, x_idx],
            upper_ci,  # mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
            lower_ci,  # mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
            color="lightgreen",
            alpha=0.5,
        )
    # Do we want to do category specific coloring?
    else:
        _ = sns.lineplot(
            x=x_new[:, x_idx],
            y=mean.numpy().flatten(),
            linewidth=2.5,
            label=label,
            color=cat_color_pal[int(label) % len(cat_color_pal)],
            ax=ax,
        )
        ax.fill_between(
            x_new[:, x_idx],
            upper_ci,  # mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
            lower_ci,  # mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
            color=cat_color_pal[int(label) % len(cat_color_pal)],
            alpha=0.5,
        )

    ax.plot(
        x_new[:, x_idx],
        samples,  # [:, :, 0].numpy().T,# "C0",
        color="dimgray",
        linewidth=0.5,
        alpha=0.25,
    )
    #     plt.close()

    ax.set(
        xlabel=(
            f"""{replace_kernel_variables(
                '['+str(x_idx)+']',
                col_names
            ).strip('[]')}"""
        )
    )
    return ax
