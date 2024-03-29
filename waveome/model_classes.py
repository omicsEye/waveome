import copy

import gpflow
import numpy as np
import tensorflow as tf

# Not currently using below because it presents warning on M1 mac
# from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from gpflow.base import RegressionData
from joblib import Parallel, delayed
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow_probability import distributions as tfd
from tqdm import tqdm

from .kernels import Empty
from .predictions import gp_predict_fun, pred_kernel_parts
from .regularization import make_folds
from .utilities import (
    calc_bic,
    find_variance_components,
    find_variance_components_tf,
    gp_likelihood_crosswalk,
    hmc_sampling,
    print_kernel_names,
    tqdm_joblib,
    variance_contributions,
)


class BaseGP(gpflow.models.SVGP):
    """Basic Gaussian process that inherits SVGP gpflow structure.

    Attributes
    ----------
    X: numpy.array
        Input array of covariates.
    Y: numpy.array
        Array containing output values.
    kernel: gpflow.kernel.Kernel
        GPflow kernel specification.
    verbose: boolean
        Flag to request more output when calling methods.

    Methods
    -------
    update_kernel_name()
        Updates the saved kernel name string based on kernel components.
    randomize_params(loc=0.0, scale=1.0, random_seed=None)
        Randomizes traininable parameters in model.
    optimize_params(
        adam_learning_rate=0.01,
        nat_gradient_gamma=0.01,
        num_opt_iter=5000,
        convergence_threshold=1e-6,
    )
        Optimizes trainable parameters in model using Adam and Natural
        Gradient methods.
    random_restart_optimize(
        num_restart=5,
        randomize_kwargs={},
        optimize_kwargs={}
    )
        Combines random initialization and optimization.
    variance_explained()
        Calculates variance explained for each additive kernel component.
    calc_metric(metric="BIC")
        Calculates the requested metric (currently only BIC) for the model.
    plot_functions(x_idx, col_names, **kwargs)
        Plots the marginal distribution of functions for a particular
        dimension.
    plot_parts(x_idx, col_names, lik=None, **kwargs)
        Plots the additive components for a model.

    Notes
    -----
    This class is the base Gaussian process class used in waveome. Extensions
    such as variational, sparse, and penalized GPs are built off of this.

    """

    def __init__(
        self,
        X: np.array,
        Y: np.array,
        mean_function: gpflow.functions.Function = gpflow.functions.Constant(
            c=0.0
        ),
        kernel: gpflow.kernels.Kernel = gpflow.kernels.SquaredExponential(
            active_dims=[0]
        ),
        verbose: bool = False,
        **svgp_kwargs,
    ):
        # Set values
        self.X = X
        self.Y = Y
        self.data = (
            tf.convert_to_tensor(X, dtype=tf.float64),
            tf.convert_to_tensor(Y, dtype=tf.float64)
        )
        self.mean_function = mean_function
        self.kernel = kernel
        self.kernel_name = ""
        self.verbose = verbose

        if hasattr(self, "num_inducing_points") is False:
            self.num_inducing_points = X.shape[0]

        # Check for missing data
        assert (
            np.isnan(X).sum() == 0
        ), "Missing values in X found. This is currently not allowed!"
        assert (
            np.isnan(Y).sum() == 0
        ), "Missing values in Y found. This is currently not allowed!"

        # Fill in information for parent class
        super().__init__(
            mean_function=mean_function,
            kernel=kernel,
            likelihood=gpflow.likelihoods.Gaussian(),
            inducing_variable=gpflow.inducing_variables.InducingPoints(X),
            **svgp_kwargs,
        )

        # Get the string for the kernel name as well
        self.update_kernel_name()

        # Freeze inducing variables for this model type
        gpflow.utilities.set_trainable(self.inducing_variable, False)

    def update_kernel_name(self):
        try:
            self.kernel_name = print_kernel_names(self.kernel, with_idx=True)
        except TypeError:
            self.kernel_name = print_kernel_names(self.kernel)
        if type(self.kernel_name) not in [list, str]:
            self.kernel_name = "+".join(list(self.kernel_name))

        return None

    def randomize_params(
        self, loc: float = 0.0, scale: float = 1.0, random_seed: int = None
    ) -> None:
        """Randomize model parameters from sampled normal distribution.

        Parameters
        ----------
        loc: float
            Mean of normal distribution to sample from.
        scale: float
            Standard deviation of normal distribution to sample from.
        random_seed: int
            Random seed for reproducibility.

        Returns
        -------
        None
            Returns randomized self.trainable_parameters values in self.

        """

        # Set random seed
        np.random.seed(random_seed)

        # Go through each trainable variable
        for p in self.trainable_parameters:
            # Sample unconstrained values from a distribution
            # Need to do special stuff for q_std (triangular fill)
            if p.name == "fill_triangular":
                sample_matrix = np.diag(
                    # np.random.exponential(
                    #   scale=scale, size=(self.X.shape[0])
                    # )
                    np.random.exponential(
                        scale=scale, size=self.num_inducing_points
                    )
                )
                try:
                    p.assign(tf.convert_to_tensor(sample_matrix[None, :, :]))
                except ValueError:
                    if self.verbose:
                        print(
                            "Error assigning random values"
                            " to fill_triangular, skipping!"
                        )

            else:
                unconstrain_vals = np.random.normal(
                    loc=loc, scale=scale, size=p.numpy().shape
                )

                # Assign those values to the trainable variable
                p.assign(p.transform_fn(unconstrain_vals))
        return None

    def optimize_params(
        self,
        adam_learning_rate=0.01,
        nat_gradient_gamma=1.0,
        num_opt_iter=20000,
        convergence_threshold=1e-3,
    ):
        """Optimize hyperparameters of model.

        Parameters
        ----------
        adam_learning_rate: float
            Learning rate for Adam optimizer.
        nat_gradient_gamma: float
            Gamma rate for Natural Gradient optimizer.
        num_opt_iter: int
            Number of iterations to perform during optimization.
        convergence_threshold: float
            Threshold to stop optimization early. Currently this is the
            difference in log likelihood between iterations.

        Returns
        -------
        None
            Self model parameters are optimized and saved in object.

        Notes
        -----
        Adam optimizer is used for all model parameters except for the
        variational components (q_mu, q_sqrt). Those are optimized using
        the Natural Gradient method. Early stopping is currently baked in.
        """
        # Source: (https://gpflow.github.io/GPflow/develop/
        # notebooks/advanced/natural_gradients.html#Natural-gradients)

        # Can we just use BFGS for "smaller" models?
        tot_params = 0
        for var in self.trainable_variables:
            num_params = np.prod(var.shape)
            tot_params += num_params
        if tot_params <= 10000:
            if self.verbose:
                print(f"Number of params: {tot_params}, using Scipy optimizer")
            # try:
            gpflow.optimizers.Scipy().minimize(
                closure=self.training_loss_closure(
                    data=self.data,
                    compile=True
                ),
                variables=self.trainable_variables,
                method="L-BFGS-B",
                options={"maxiter": num_opt_iter}
            )
            # except TypeError:
            #     for param in self.parameters:
            #         param = tf.cast(param, tf.float32)
            #     gpflow.optimizers.Scipy().minimize(
            #         closure=self.training_loss_closure(
            #             data=(
            #                 tf.cast(self.data[0], tf.float32),
            #                 tf.cast(self.data[1], tf.float32)
            #             ),
            #             compile=True
            #         ),
            #         variables=self.trainable_variables,
            #         method="L-BFGS-B",
            #         options={"maxiter": num_opt_iter}
            #     )

            return None

        # Stop Adam from optimizing the variational parameters
        gpflow.set_trainable(self.q_mu, False)
        gpflow.set_trainable(self.q_sqrt, False)

        # Create the optimize_tensors for VGP with natural gradients
        adam_opt = Adam(learning_rate=adam_learning_rate)
        natgrad_opt = gpflow.optimizers.NaturalGradient(
            gamma=nat_gradient_gamma
        )

        # Compile loss closure based on training data
        compiled_loss = self.training_loss_closure(
            data=(self.X, self.Y), compile=True
        )

        @tf.function
        def split_optimization_step():
            adam_opt.minimize(
                compiled_loss,
                var_list=self.trainable_variables,
            )
            natgrad_opt.minimize(
                compiled_loss, var_list=[(self.q_mu, self.q_sqrt)]
            )

        loss_list = []
        # Now optimize the parameters
        for i in range(num_opt_iter):
            
            # Save previous values
            previous_values = gpflow.utilities.deepcopy(
                gpflow.utilities.parameter_dict(self)
            )

            try:
                split_optimization_step()
            except tf.errors.InvalidArgumentError:
                print(
                    "Reached invalid step in optimization,"
                    " returning previous step."
                )
                gpflow.utilities.multiple_assign(self, previous_values)
                break
            if i % 5 == 0:
                cur_loss = self.training_loss((self.X, self.Y))
                loss_list.append(cur_loss)
                if i % 500 == 0 and self.verbose:
                    print(f"Round {i} training loss: {cur_loss}")

            if (
                len(loss_list) > 1
                and loss_list[-2] - loss_list[-1] < convergence_threshold
            ):
                if self.verbose:
                    print(
                        f"Optimization converged - stopping early (round {i})"
                    )
                break

        if i == (num_opt_iter - 1):
            print(f"Optimization not converged after {i+1} rounds")

        return None

    def random_restart_optimize(
        self, num_restart=5, randomize_kwargs={}, optimize_kwargs={}
    ):
        """Randomize and optimize hyperparameters a certain number of times
        to search space.

        Parameters
        ----------
        num_restart: int
        randomize_kwargs: dict
        optimize_kwargs: dict

        Returns
        -------
        None


        """
        # Set initial log likelihood to track during restarts
        max_ll = -np.inf
        best_variables = {}

        for i in range(num_restart):
            if self.verbose:
                print(f"Random restart {i+1}")
            # Randomize parameters
            self.randomize_params(**randomize_kwargs)

            # Optimize parameters
            self.optimize_params(**optimize_kwargs)

            # Check if log likelihood is better
            cur_max_ll = self.maximum_log_likelihood_objective(
                data=(self.X, self.Y)
            )
            if cur_max_ll > max_ll:
                max_ll = cur_max_ll
                best_variables = gpflow.utilities.deepcopy(
                    gpflow.utilities.parameter_dict(self)
                )
                if self.verbose:
                    print("Found better parameters!")

        # Set trainable variables to the best found
        gpflow.utilities.multiple_assign(self, best_variables)

        return None

    def get_variance_explained(self):
        """Calculates variance explained for each additive kernel component.

        Arguments
        ---------
        None

        Returns
        -------
        None
            Variance contributions in self.variance_explained
        """
        self.variance_explained = list(
            variance_contributions(
                self, k_names=self.kernel_name, lik=self.likelihood.name
            )
        )

        # return self.variance_explained
        return None

    def calc_metric(self, metric="BIC"):
        assert metric == "BIC", "Only BIC currently allowed."
        if metric == "BIC":
            return calc_bic(
                loglik=self.log_posterior_density(self.data),
                n=self.X.shape[0],
                k=len(self.trainable_parameters),
            )

    def plot_functions(self, x_idx, col_names, **kwargs):
        return gp_predict_fun(
            self, x_idx, col_names, X=self.X, Y=self.Y, **kwargs
        )

    def plot_parts(self, x_idx, col_names, lik=None, **kwargs):
        if lik is None:
            lik = self.likelihood
        return pred_kernel_parts(
            self,
            x_idx=x_idx,
            # unit_idx=unit_idx,
            col_names=col_names,
            lik=lik,
            **kwargs,
        )


class VarGP(BaseGP):
    """Variational Gaussian process that inherits SVGP gpflow structure.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(
        self,
        X,
        Y,
        mean_function=gpflow.functions.Constant(c=0.0),
        kernel=gpflow.kernels.SquaredExponential(active_dims=[0]),
        likelihood="gaussian",
        variational_priors=True,
        verbose=False,
        **basegp_kwargs,
    ):
        # Set values
        self.X = X
        self.Y = Y
        self.mean_function = mean_function
        self.kernel = kernel
        self.verbose = verbose

        # Fill in information for parent class
        super().__init__(
            X=X,
            Y=Y,
            mean_function=mean_function,
            kernel=kernel,
            verbose=verbose,
            **basegp_kwargs,
        )

        # Freeze inducing variables for this model type
        gpflow.utilities.set_trainable(self.inducing_variable, False)

        # Now set likelihood based on argument
        if type(likelihood) is str:
            self.likelihood = gp_likelihood_crosswalk(likelihood)
        elif "gpflow.likelihoods." in str(type(likelihood)):
            self.likelihood = likelihood
        else:
            raise (
                """Unknown likelihood requested. Either string or
                gpflow.likelihood allowed"""
            )

        # Set variational priors if requested
        if variational_priors is True:
            self.q_mu.prior = tfd.Normal(loc=0.0, scale=100.0)
            self.q_sqrt.prior = tfd.HalfNormal(scale=100.0)


class SparseGP(BaseGP):
    """Sparse Gaussian process that inherits SVGP gpflow structure.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(
        self,
        X,
        Y,
        mean_function=gpflow.functions.Constant(c=0.0),
        kernel=gpflow.kernels.SquaredExponential(active_dims=[0]),
        num_inducing_points=500,
        train_inducing=True,
        random_points=True,
        random_seed=None,
        verbose=False,
        **basegp_kwargs,
    ):
        # Set values
        self.X = X
        self.Y = Y
        self.mean_function = mean_function
        self.kernel = kernel
        self.verbose = verbose
        self.num_inducing_points = num_inducing_points

        # Check inducing point size compared to training data
        # If more than we need then we set to dataset size and don't train
        if num_inducing_points > X.shape[0]:
            if verbose:
                print(
                    "Number of inducing points requested "
                    f"({num_inducing_points}) greater than "
                    f"original data ({X.shape[0]})"
                )
            num_inducing_points = X.shape[0]
            train_inducing = False

            self.num_inducing_points = num_inducing_points

        # Fill in information for parent class
        super().__init__(
            X=X,
            Y=Y,
            mean_function=mean_function,
            kernel=kernel,
            verbose=verbose,
            q_mu=np.zeros(num_inducing_points)[:, None],
            q_sqrt=np.eye(num_inducing_points)[None, :, :],
            **basegp_kwargs,
        )

        # Set inducing points
        if random_points is True:
            if random_seed is not None:
                np.random.seed(random_seed)
            obs_idx = np.random.choice(
                a=np.arange(X.shape[0]),
                size=num_inducing_points,
                replace=False,
            )
        else:
            obs_idx = np.arange(num_inducing_points)

        self.inducing_variable = gpflow.inducing_variables.InducingPoints(
            X[obs_idx, :].copy()
        )
        gpflow.utilities.set_trainable(self.inducing_variable, train_inducing)


class PenalizedGP(BaseGP):
    """Penalized Gaussian process via priors on variance hyperparameters.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(
        self,
        X,
        Y,
        mean_function=gpflow.functions.Constant(c=0.0),
        kernel=gpflow.kernels.SquaredExponential(active_dims=[0]),
        penalization_factor=1.0,
        verbose=False,
        **basegp_kwargs,
    ):
        # Parent class
        super().__init__(
            X=X,
            Y=Y,
            mean_function=mean_function,
            kernel=kernel,
            verbose=verbose,
            **basegp_kwargs,
        )

        # Set initial factor given
        self.set_penalization_factor(penalization_factor)

        # Set unit col as none for now
        self.unit_col = None
        self.penalization_search_results = None

    def maximum_log_likelihood_objective(self, data):
        model_fit = self.elbo(data)
        model_var = (
            data[0].shape[0]
            * self.penalization_factor
            * find_variance_components_tf(self.kernel)
        )
        return model_fit - model_var

    def set_penalization_factor(self, penalization_factor, use_prior=False):
        self.penalization_factor = penalization_factor

        if use_prior:
            # Set prior on kernel variance terms
            if penalization_factor > 0:
                # hyperprior = tfd.Exponential(
                #     rate=gpflow.utilities.to_default_float(penalization_factor),
                #     name="hyperprior"
                # )
                prior = tfd.Exponential(
                    # rate=hyperprior.sample()
                    rate=gpflow.utilities.to_default_float(penalization_factor)
                )
            else:
                prior = None
            for key, val in gpflow.utilities.parameter_dict(self).items():
                if "kernel" in key and "variance" in key:
                    val.prior = prior

    def penalization_search(
        self,
        penalization_factor_list=np.exp(np.linspace(0, 10, 5)),
        k_fold=3,
        fit_best=True,
        max_jobs=-1,
        show_progress=True,
        parallel_object=None,
        randomization_options={},
        optimization_options={},
        random_seed=None,
    ):
        # Split training data into k-folds
        folds = make_folds(self.X, self.unit_col, k_fold, random_seed)

        # Set random seed in randomization options if not defined
        if "random_seed" not in randomization_options.keys():
            randomization_options["random_seed"] = random_seed

        # Figure out total combinations of fold x factor
        ff_idx = np.array(
            np.meshgrid(
                np.arange(len(penalization_factor_list)), np.arange(len(folds))
            )
        ).T.reshape(-1, 2)

        # Fit combinations in parallel if possible
        def parallel_fit(pf, holdout_fold, holdout_index):
            # temp_model = copy.deepcopy(self)
            temp_model = PenalizedGP(
                X=np.delete(self.X, holdout_fold, axis=0),
                Y=np.delete(self.Y, holdout_fold, axis=0),
                kernel=copy.deepcopy(self.kernel),
                penalization_factor=pf,
                verbose=self.verbose,
            )
            # temp_model.set_penalization_factor(pf)
            holdout_X = self.X[holdout_fold]
            holdout_Y = self.Y[holdout_fold]
            # temp_model.X = np.delete(self.X, holdout_fold, axis=0)
            # temp_model.Y = np.delete(self.Y, holdout_fold, axis=0)
            temp_model.randomize_params(**randomization_options)
            temp_model.optimize_params(**optimization_options)
            holdout = np.mean(
                temp_model.predict_log_density(data=(holdout_X, holdout_Y))
            )

            # self.set_penalization_factor(pf)
            # holdout_X = self.X[holdout_fold]
            # holdout_Y = self.Y[holdout_fold]
            # self.X = np.delete(self.X, holdout_fold, axis=0)
            # self.Y = np.delete(self.Y, holdout_fold, axis=0)
            # self.randomize_params()
            # self.optimize_params()
            # holdout = np.mean(
            #     self.predict_log_density(
            #         data=(holdout_X, holdout_Y)
            #     )
            # )

            return np.array([pf, holdout_index, holdout])

        # Do these in parallel if possible
        # Pass in parallel API already constructed if possible
        if parallel_object is not None:
            parallel_results = parallel_object(
                delayed(parallel_fit)(
                    pf=penalization_factor_list[i[0]],
                    holdout_fold=folds[i[1]],
                    holdout_index=i[1],
                )
                for i in ff_idx
            )

        else:
            if show_progress:
                with tqdm_joblib(
                    tqdm(desc="Penalization search", total=ff_idx.shape[0])
                ) as _:
                    parallel_results = Parallel(n_jobs=max_jobs)(
                        delayed(parallel_fit)(
                            pf=penalization_factor_list[i[0]],
                            holdout_fold=folds[i[1]],
                            holdout_index=i[1],
                        )
                        for i in ff_idx
                    )
            else:
                parallel_results = Parallel(n_jobs=max_jobs)(
                    delayed(parallel_fit)(
                        pf=penalization_factor_list[i[0]],
                        holdout_fold=folds[i[1]],
                        holdout_index=i[1],
                    )
                    for i in ff_idx
                )

        parallel_results = np.vstack(parallel_results)
        self.penalization_search_results = parallel_results

        # Find best penalization factor
        max_val = -np.inf
        max_factor = -np.inf
        for factor in penalization_factor_list:
            cur_val = parallel_results[
                parallel_results[:, 0] == factor, 2
            ].mean()
            if cur_val > max_val:
                max_factor = factor
                max_val = cur_val
        best_factor = max_factor

        # Set penalization factor to zero if still -np.inf
        if max_factor == -np.inf:
            if self.verbose:
                print("Search error, returning no penalization")
            max_val = 0.0

        if self.verbose:
            print(f"Best penalization factor found from search: {best_factor}")

        # Fit full model with best penalization value found
        if fit_best:
            self.set_penalization_factor(best_factor)
            self.randomize_params(**randomization_options)
            self.optimize_params(**optimization_options)

    # def penalized_optimization(self, holdout_X, holdout_Y):
        
    #     # Set penalization factor to be a trainable parameter
    #     self.pf = gpflow.Parameter(
    #         value=self.penalization_factor,
    #         transform=gpflow.utilities.positive(),
    #         name="pf"
    #     )
    #     # self.set_penalization_factor(self.pf)

    #     for i in range(5):
    #         # Optimize hyperparameters given training data
    #         gpflow.utilities.set_trainable(self.pf, False)
    #         print("Optimizing hyperparameters")
    #         self.optimize_params(num_opt_iter=100)

    #         # Optimize penalization factor given holdout data
    #         gpflow.utilities.set_trainable(self.pf, True)
    #         print("Optimizing penalization factor")
    #         gpflow.optimizers.Scipy().minimize(
    #             closure=self.training_loss_closure(
    #                 (holdout_X, holdout_Y)
    #             ),
    #             variables=[(self.trainable_variables[0])]
    #         )
    #         print(self.pf)

    #     return None

    def cut_kernel_components(self, var_cutoff: float = 0.001):
        """Prune out kernel components with small variance parameters

        Parameters
        ----------
        model
        var_cutoff

        Returns
        -------
        GPflow model
        """

        # Return empty model object if none passed in
        if self is None:
            return self

        # Get variance components for each additive kernel part
        var_parts = find_variance_components(self.kernel, sum_reduce=False)
        var_flag = np.where(var_parts >= var_cutoff)[0]

        # Copy model
        # new_model = gpflow.utilities.deepcopy(model)

        # Figure out which ones should be kept and sum if needed
        if len(var_flag) > 1:
            self.kernel = gpflow.kernels.Sum(
                [self.kernel.kernels[i] for i in var_flag]
            )
        elif len(var_flag) == 1:
            if len(var_parts) > 1:
                self.kernel = self.kernel.kernels[var_flag[0]]
            else:
                self.kernel = self.kernel
        else:
            self.kernel = Empty()

        # Also make sure we only keep base variances that remain
        # This will not be used post refactor of code base
        if hasattr(self, "base_variances"):
            if self.base_variances is not None:
                self.base_variances = self.base_variances[var_flag]

        return None


class PSVGP(PenalizedGP, SparseGP, VarGP):
    """Combine all of the Gaussian process types into the main entry point.

    Attributes
    ----------
    X: numpy.ndarray
    Y: numpy.ndarray
    mean_function: gpflow.functions.Function
    kernel: gpflow.kernels.Kernel
    verbose: bool
    penalized_options: dict
    sparse_options: dict
    variational_options: dict

    Methods
    -------

    """

    def __init__(
        self,
        X,
        Y,
        mean_function=gpflow.functions.Constant(c=0.0),
        kernel=gpflow.kernels.SquaredExponential(active_dims=[0]),
        verbose=False,
        penalized_options={},
        sparse_options={},
        variational_options={},
    ):
        super().__init__(
            X=X,
            Y=Y,
            mean_function=mean_function,
            kernel=kernel,
            verbose=verbose,
            **penalized_options,
            **sparse_options,
            **variational_options,
        )


# class PSVGPMC(gpflow.models.SGPMC):
#     """Sparse GP with MC capabilities."""

#     def __init__(
#         self,
#         X,
#         Y,
#         kernel=gpflow.kernels.SquaredExponential(active_dims=[0]),
#         likelihood=gpflow.likelihoods.Gaussian(),
#         inducing_variable=None,
#         mean_function=gpflow.functions.Constant(),
#         verbose=False,
#         **kwargs,
#     ):
#         # Set values
#         self.X = X
#         self.Y = Y
#         self.data = (tf.convert_to_tensor(X), tf.convert_to_tensor(Y))
#         self.mean_function = mean_function
#         self.kernel = kernel
#         self.kernel_name = ""
#         self.verbose = verbose
#         self.penalization_term = gpflow.Parameter(
#             1., transform=tfp.bijectors.Exp()
#         )

#         # Fill in information for parent class
#         super().__init__(
#             data=self.data,
#             mean_function=mean_function,
#             kernel=kernel,
#             likelihood=likelihood,
#             inducing_variable=inducing_variable
#             # inducing_variable=gpflow.inducing_variables.InducingPoints(data[0]),
#             # **svgpmc_kwargs,
#         )

#         if hasattr(self, "num_inducing_points") is False:
#             self.num_inducing_points = X.shape[0]

#         # Check for missing data
#         assert (
#             np.isnan(X).sum() == 0
#         ), "Missing values in X found. This is currently not allowed!"
#         assert (
#             np.isnan(Y).sum() == 0
#         ), "Missing values in Y found. This is currently not allowed!"

#         # Get the string for the kernel name as well
#         # self.update_kernel_name()

#         # # Freeze inducing variables for this model type
#         # gpflow.utilities.set_trainable(self.inducing_variable, False)

#     def log_likelihood_lower_bound(self):
#         X_data, Y_data = self.data
#         fmean, fvar = self.predict_f(X_data, full_cov=False)
#         print( find_variance_components_tf(
#                 self.kernel,
#                 sum_reduce=True,
#             ))
#         pen_factor = tf.multiply(
#             self.penalization_term,
#             find_variance_components_tf(
#                 self.kernel,
#                 sum_reduce=True,
#             )
#         )
#         var_exp = tf.reduce_sum(
#             self.likelihood.variational_expectations(
#                 X_data,
#                 fmean,
#                 fvar,
#                 Y_data
#             )
#         )
#         return var_exp #tf.subtract(var_exp, pen_factor)

#     def sample_posterior(
#         self,
#         burn_in=500,
#         samples=1000,
#         random_seed=None,
#         step_size=0.01,
#         accept_prob=0.9,
#         num_adaptation_steps=100
#     ):
#         return hmc_sampling(
#             self,
#             burn_in=burn_in,
#             samples=samples,
#             random_seed=random_seed,
#             step_size=step_size,
#             accept_prob=accept_prob,
#             num_adaptation_steps=num_adaptation_steps
#         )
