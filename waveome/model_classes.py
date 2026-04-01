import copy
from socket import has_dualstack_ipv6

import gpflow
import gpflow.inducing_variables as iv
import numpy as np
import tensorflow as tf
from gpflow.utilities import deepcopy
from joblib import Parallel, delayed
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow_probability import distributions as tfd
from tqdm import tqdm

from .kernels import Categorical, Empty
from .predictions import gp_predict_fun, pred_kernel_parts
from .regularization import make_folds
from .utilities import (
    calc_bic,
    calc_feature_importance_components,
    convert_data_to_tensors,
    find_variance_components,
    find_variance_components_tf,
    gp_likelihood_crosswalk,
    print_kernel_names,
    search_through_kernel_list_,
    tqdm_joblib,
)

f64 = gpflow.utilities.to_default_float


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
    randomize_params(loc=0.0, scale=10.0, random_seed=None)
        Randomizes traininable parameters in model.
    optimize_params(
        adam_learning_rate=0.001,
        nat_gradient_gamma=0.001,
        num_opt_iter=5000,
        convergence_threshold=0.1,
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
        mean_function: gpflow.functions.Function = (gpflow.functions.Constant(c=0.0)),
        kernel: gpflow.kernels.Kernel = (
            gpflow.kernels.SquaredExponential(active_dims=[0])
        ),
        verbose: bool = False,
        num_latent_gps=1,
        dtype=None,
        **svgp_kwargs,
    ):

        # Set up inducing variables
        if num_latent_gps == 1:
            inducing_variable = iv.InducingPoints(X)
        else:
            inducing_variable = iv.SeparateIndependentInducingVariables(
                [
                    gpflow.inducing_variables.InducingPoints(X_)
                    for X_ in [X.copy() for _ in range(num_latent_gps)]
                ]
            )

        # Fill in information for parent class
        super().__init__(
            mean_function=deepcopy(mean_function),
            kernel=deepcopy(kernel),
            likelihood=gpflow.likelihoods.Gaussian(),
            inducing_variable=inducing_variable,
            num_latent_gps=num_latent_gps,
            **svgp_kwargs,
        )

        # Set values
        # Comment this out for now to save memory
        # self.X = X.astype(gpflow.default_float())
        # self.Y = Y.astype(gpflow.default_float())
        self.data = (
            tf.reshape(
                tensor=tf.convert_to_tensor(
                    X,
                    dtype=gpflow.default_float(),
                    # dtype=tf.float64 if dtype == "float64" else tf.float32
                ),
                shape=(X.shape),
            ),
            tf.reshape(
                tensor=tf.convert_to_tensor(
                    Y,
                    dtype=gpflow.default_float(),
                    # dtype=tf.float64 if dtype == "float64" else tf.float32
                ),
                shape=(Y.shape),
            ),
        )
        # self.mean_function = deepcopy(mean_function)
        # self.kernel = deepcopy(kernel)
        self.kernel_name = ""
        self.verbose = verbose
        self.optimizer = None
        self.num_trainable_params = np.nan
        self.num_latent_gps = num_latent_gps

        if hasattr(self, "num_inducing_points") is False:
            self.num_inducing_points = X.shape[0]

        # Rest variational parameters because num_latent_gps isn't respected
        self._init_variational_parameters(
            num_inducing=self.num_inducing_points, q_mu=None, q_sqrt=None, q_diag=False
        )

        # Check for missing data
        assert (
            np.isnan(X).sum() == 0
        ), "Missing values in X found. This is currently not allowed!"
        assert (
            np.isnan(Y).sum() == 0
        ), "Missing values in Y found. This is currently not allowed!"

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
                    ).astype(gpflow.default_float())
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
                ).astype(gpflow.default_float())

                # Assign those values to the trainable variable
                p.assign(p.transform_fn(unconstrain_vals))
        return None

    def optimize_params(
        self,
        adam_learning_rate=0.1,
        adam_decay_rate=0.96,
        nat_gradient_gamma=0.1,
        num_opt_iter=50000,
        minibatch_size=None,
        convergence_threshold=1e-9,
        optimizer="adam/gradient",
        data=None,
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

        # Reset graph
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        gpflow.utilities.reset_cache_bijectors(self)

        # Can we just use BFGS for "smaller" models? Exclude variational params
        # tot_params = 0
        # for var_set in [
        #     self.mean_function.trainable_variables,
        #     self.kernel.trainable_variables
        # ]:
        #     for var in var_set:
        #         num_params = np.prod(var.shape)
        #         tot_params += num_params
        if np.isnan(self.num_trainable_params):
            tot_params = 0
            for var in self.trainable_variables:
                if "fill_triangular" in var.name:
                    num_params = tf.shape(var)[0]
                else:
                    num_params = np.prod(tf.shape(var))
                tot_params += num_params
            self.num_trainable_params = tot_params

        if (
            self.num_trainable_params <= 5000 and optimizer is None
        ) or optimizer == "scipy":
            if self.verbose:
                print(
                    f"Number of params: {self.num_trainable_params},",
                    " using Scipy optimizer",
                )
            self.optimizer = "scipy"

            optimizer = gpflow.optimizers.Scipy()
            opt_options = {
                "maxiter": num_opt_iter,
                "maxfun": num_opt_iter,
                #     "ftol": convergence_threshold,
                #     "maxcor": 100
            }

            # Make sure we freeze inducing points if the kernel is Constant()
            # otherwise we get unconnected gradient error
            # issue: https://github.com/GPflow/GPflow/issues/1600
            if self.kernel.name == "constant":
                gpflow.utilities.set_trainable(self.inducing_variable, False)

            for attempt in range(5):
                try:
                    optimizer.minimize(
                        closure=self.training_loss_closure(
                            data=data,
                            # compile=False
                            # Need to not compile with Scipy optimizer
                        ),
                        variables=self.trainable_variables,
                        method="L-BFGS-B",
                        options=opt_options,
                    )
                    break
                except Exception as e:
                    if self.verbose:
                        print(f"Attempt {attempt+1} - Scipy exception: {e}")
                    for param in self.parameters:
                        param = tf.cast(param, gpflow.default_float())

            return None

        if (
            self.num_trainable_params > 5000 and optimizer is None
        ) or optimizer == "adam/gradient":
            # Set optimizer otherwise
            self.optimizer = "adam/gradient"

            # Stop Adam from optimizing the variational parameters
            gpflow.set_trainable(self.q_mu, False)
            gpflow.set_trainable(self.q_sqrt, False)

            # Create the optimize_tensors for VGP with natural gradients
            adam_opt = Adam(learning_rate=adam_learning_rate)
            natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=nat_gradient_gamma)

            @tf.function
            def split_optimization_step():
                adam_opt.minimize(
                    compiled_loss,
                    var_list=self.trainable_variables,
                )
                natgrad_opt.minimize(compiled_loss, var_list=[(self.q_mu, self.q_sqrt)])

        elif optimizer == "adam":
            self.optimizer = "adam"
            adam_opt = Adam(learning_rate=adam_learning_rate)

            @tf.function
            def split_optimization_step():
                adam_opt.minimize(
                    compiled_loss,
                    var_list=self.trainable_variables,
                )

        else:
            ValueError(
                "Unknown optimizer selected!",
                " Current options: ['scipy', 'adam', 'adam/gradient', None]",
            )

        # Set up loss based on minibatching or not
        if minibatch_size is not None:
            data_minibatch = (
                tf.data.Dataset.from_tensor_slices(data)
                .prefetch(tf.data.AUTOTUNE)
                .repeat()
                .shuffle(100)
                .batch(minibatch_size)
            )
            # data_minibatch_it = iter(data_minibatch)
            compiled_loss = self.training_loss_closure(iter(data_minibatch))
        else:
            # Compile loss closure based on training data
            compiled_loss = self.training_loss_closure(data=data, compile=True)

        loss_list = []

        # Save initial values
        previous_values = gpflow.utilities.deepcopy(
            gpflow.utilities.parameter_dict(self)
        )
        # Now optimize the parameters
        for i in range(num_opt_iter):

            # optimization step
            try:
                split_optimization_step()
            except tf.errors.InvalidArgumentError:
                print(
                    "Reached invalid step in optimization," " returning previous step."
                )
                gpflow.utilities.multiple_assign(self, previous_values)
                break

            # Checkpoint!
            if i % 100 == 0:
                # Save previous values
                previous_values = gpflow.utilities.deepcopy(
                    gpflow.utilities.parameter_dict(self)
                )

                # Calculate loss
                cur_loss = self.training_loss(data)
                # Check to make sure we haven't gone off the rails
                if np.isnan(cur_loss):
                    print("NaN loss, something went wrong - stopping!")
                    break
                else:
                    loss_list.append(cur_loss)

                # Update adam learning rate (decay)
                # Similar to (
                #   www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
                # )
                if i % 500 == 0:
                    adam_opt.learning_rate = adam_learning_rate * adam_decay_rate ** (
                        i / 500
                    )

                    # if self.optimizer == "adam/gradient":
                    #     natgrad_opt.gamma = (
                    #         nat_gradient_gamma *
                    #         adam_decay_rate ** (i / 500)
                    #     )

                    # Print output if requested
                    if self.verbose:
                        print(f"Round {i} training loss: {cur_loss}")
                        print(f"New learning rate: {adam_opt.learning_rate}")
                        if self.optimizer == "adam/gradient":
                            print(f"New gamma rate: {natgrad_opt.gamma}")

                # Check to see if we have reached convergence
                if (
                    len(loss_list) > 1
                    and loss_list[-2] - loss_list[-1] < convergence_threshold
                ):
                    if self.verbose:
                        print(f"Optimization converged - stopping early (round {i})")
                    break

        # If we have reached the end of iterations and still
        # not converged then...
        if i == (num_opt_iter - 1):
            if self.verbose:
                print(f"Optimization not converged after {i+1} rounds")

        return None

    def random_restart_optimize(
        self, data=None, num_restart=5, randomize_kwargs={}, optimize_kwargs={}
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

            # Increment random seed if one is given
            if "random_seed" in randomize_kwargs:
                if randomize_kwargs["random_seed"] is None:
                    randomize_kwargs["random_seed"] = i
                else:
                    randomize_kwargs["random_seed"] += 1

            # Randomize parameters
            self.randomize_params(**randomize_kwargs)

            # Optimize parameters
            self.optimize_params(**optimize_kwargs, data=data)

            # Check if log likelihood is better
            cur_max_ll = self.maximum_log_likelihood_objective(data=data)
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

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        """
        Compute mean and (co)variance of latent function at Xnew.
        Automatically casts Xnew to the default float type.
        """
        Xnew = tf.cast(Xnew, gpflow.default_float())
        return super().predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )

    def predict_y(self, Xnew, full_cov=False, full_output_cov=False):
        """
        Compute mean and (co)variance of predictive distribution at Xnew.
        Automatically casts Xnew to the default float type.
        """
        Xnew = tf.cast(Xnew, gpflow.default_float())
        return super().predict_y(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )

    def get_feature_importances(self, data=None, return_value="log_bf"):
        """Calculates feature importance for each additive kernel component.

        Arguments
        ---------
        data: tuple
            Tuple of (X, Y) data to use for calculating feature importance.
        return_value: str
            Value to return for each component. Options are:
            "log_bf" (default - log bayes factor), "statistic" (chi-squared),
            "de" (deviance explained). See calc_feature_importance_components
            for more details.

        Returns
        -------
        None
            Feature importances in self.feature_importances
        """

        # var_list = calc_rsquare(self, data=data)
        importance_list = calc_feature_importance_components(
            model=self, data=data, return_value=return_value
        )

        # Fix ListWrapper issue with Tensorflow tensors
        self.feature_importances = list(importance_list)

        return None

    def calc_metric(self, data=None, metric="BIC"):
        assert metric == "BIC", "Only BIC currently allowed."
        if metric == "BIC":
            return calc_bic(
                loglik=self.log_posterior_density(data),
                n=data[0].shape[0],  # self.X.shape[0],
                k=len(self.trainable_parameters),
            )

    def plot_functions(self, x_idx, col_names, data=None, **kwargs):
        return gp_predict_fun(
            self,
            x_idx=x_idx,
            col_names=col_names,
            X=data[0] if isinstance(data[0], np.ndarray) else data[0].numpy(),
            Y=data[1] if isinstance(data[1], np.ndarray) else data[1].numpy(),
            **kwargs,
        )

    def plot_parts(
        self, x_idx, col_names, data=None, lik=None, unit_idx=None, **kwargs
    ):
        if lik is None:
            lik = self.likelihood
        return pred_kernel_parts(
            self,
            x_idx=x_idx,
            col_names=col_names,
            var_explained=self.feature_importances,
            lik=lik,
            data=data,
            unit_idx=unit_idx,
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
        num_latent_gps=1,
        scale_value=None,
        # variational_priors=True,
        verbose=False,
        **basegp_kwargs,
    ):
        # # Set values
        # # self.X = X
        # # self.Y = Y
        # self.mean_function = deepcopy(mean_function)
        # self.kernel = deepcopy(kernel)
        # self.verbose = verbose

        # Fill in information for parent class
        super().__init__(
            X=X,
            Y=Y,
            mean_function=deepcopy(mean_function),
            kernel=deepcopy(kernel),
            num_latent_gps=num_latent_gps,
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

        # Set scale value for likelihood
        if scale_value is not None:
            self.likelihood.scale = scale_value

        # Set variational priors if requested
        # TODO: Need to figure out why priors lead to tf.float64 vs float32
        # issues with scipy optimizer
        # if variational_priors is True:
        #     self.q_mu.prior = tfd.Normal(loc=f64(0), scale=f64(100))
        #     self.q_sqrt.prior = tfd.HalfNormal(scale=f64(100))


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

        # Fill in information for parent class
        super().__init__(
            X=X,
            Y=Y,
            mean_function=deepcopy(mean_function),
            kernel=deepcopy(kernel),
            verbose=verbose,
            q_mu=np.zeros(num_inducing_points)[:, None],
            q_sqrt=np.eye(num_inducing_points)[None, :, :],
            **basegp_kwargs,
        )

        # Set values
        # # self.X = X
        # # self.Y = Y
        # self.mean_function = deepcopy(mean_function)
        # self.kernel = deepcopy(kernel)
        # self.verbose = verbose
        self.num_inducing_points = num_inducing_points

        # Check inducing point size compared to training data
        # If more than we need then we set to dataset size and don't train
        if num_inducing_points >= X.shape[0]:
            if verbose:
                print(
                    "Number of inducing points requested "
                    f"({num_inducing_points}) greater than or equal to "
                    f"original data size ({X.shape[0]})"
                )
            num_inducing_points = X.shape[0]
            train_inducing = False
            random_points = False

            self.num_inducing_points = num_inducing_points

        # Set inducing points
        if random_points is True:
            if random_seed is not None:
                np.random.seed(random_seed)
            obs_idx = np.random.choice(
                a=np.arange(X.shape[0]),
                size=num_inducing_points,
                replace=False,
            )

            # Choose subset for random inducing points
            sub_X = (
                X.numpy()[obs_idx, :].copy()
                if tf.is_tensor(X)
                else X[obs_idx, :].copy()
            )
            if self.num_latent_gps == 1:
                self.inducing_variable = iv.InducingPoints(sub_X)
            else:
                self.inducing_variable = iv.SeparateIndependentInducingVariables(
                    [
                        gpflow.inducing_variables.InducingPoints(X_)
                        for X_ in [sub_X.copy() for _ in range(self.num_latent_gps)]
                    ]
                )

            # Reset variational parameters
            self._init_variational_parameters(
                num_inducing=self.num_inducing_points,
                q_mu=None,
                q_sqrt=None,
                q_diag=False,
            )

        # Train inducing points if subset otherwise freeze
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
        num_latent_gps=1,
        penalization_factor=1.0,
        verbose=False,
        **basegp_kwargs,
    ):
        # Parent class
        super().__init__(
            X=X,
            Y=Y,
            mean_function=deepcopy(mean_function),
            kernel=deepcopy(kernel),
            num_latent_gps=num_latent_gps,
            verbose=verbose,
            **basegp_kwargs,
        )

        # Set initial factor given
        self.set_penalization_factor(penalization_factor)

        # Set unit col as none for now
        self.unit_col = None
        self.penalization_search_results = None

    def maximum_log_likelihood_objective(self, data, use_factor=False):
        # Returns log likelihood evidence lower bound (dependent on N)
        model_fit = self.elbo(data)

        # I cannot seem to get the penalization factor to work
        # so instead I will use Exponential priors on variance terms
        if use_factor:
            model_var = (  # tf.math.log(
                # data[0].shape[0] *
                self.penalization_factor
                * find_variance_components_tf(self.kernel)
            )
            # model_var *= data[0].shape[0]
            # out_fit = tf.math.log(tf.math.exp(model_fit) - model_var)
            # print(f"{model_fit=}, {model_var=}")
            out_fit = model_fit - model_var
        else:
            out_fit = model_fit
        return out_fit

    def set_penalization_factor(self, penalization_factor, use_prior=True):
        self.penalization_factor = gpflow.utilities.to_default_float(
            penalization_factor
        )

        if use_prior:
            # Set prior on kernel variance terms
            if penalization_factor > 0:
                # hyperprior = tfd.Exponential(
                #     rate=gpflow.utilities.to_default_float(penalization_factor),
                #     name="hyperprior"
                # )

                # Exponential prior
                # prior = tfd.Exponential(
                #     # rate=hyperprior.sample()
                #     rate=gpflow.utilities.to_default_float(penalization_factor)
                # )

                # Horseshoe prior
                prior = tfd.Horseshoe(
                    scale=gpflow.utilities.to_default_float(1.0 / penalization_factor)
                )
            else:
                prior = None
            for key, val in gpflow.utilities.parameter_dict(self).items():
                if "kernel" in key and "variance" in key:
                    val.prior = prior

    def penalization_search(
        self,
        data=None,
        # penalization_factor_list=np.exp(np.linspace(0, 10, 5)),
        penalization_factor_list=[0.0, 1.0, 10.0, 100.0],
        k_fold=3,
        fit_best=True,
        max_jobs=-1,
        show_progress=True,
        parallel_object=None,
        randomization_options={},
        optimization_options={},
        random_seed=None,
        num_restart=5,
        selection_type="se",
    ):

        # Check data types
        X = data[0] if isinstance(data[0], np.ndarray) else data[0].numpy()
        Y = data[1] if isinstance(data[1], np.ndarray) else data[1].numpy()

        # Split training data into k-folds
        folds = make_folds(X, self.unit_col, k_fold, random_seed)

        # Set random seed in randomization options if not defined
        if "random_seed" not in randomization_options.keys():
            randomization_options["random_seed"] = random_seed

        # Figure out total combinations of fold x factor
        ff_idx = np.array(
            np.meshgrid(np.arange(len(penalization_factor_list)), np.arange(len(folds)))
        ).T.reshape(-1, 2)

        # Fit combinations in parallel if possible
        def parallel_fit(pf, data, holdout_fold, holdout_index):
            temp_model = copy.deepcopy(self)
            training_data = convert_data_to_tensors(
                X=np.delete(X, holdout_fold, axis=0),
                Y=np.delete(Y, holdout_fold, axis=0),
            )
            holdout_X = X[holdout_fold]
            holdout_Y = Y[holdout_fold]
            temp_model.random_restart_optimize(
                data=training_data,
                randomize_kwargs=randomization_options,
                optimize_kwargs=optimization_options,
                num_restart=num_restart,
            )
            holdout = np.mean(
                temp_model.predict_log_density(data=(holdout_X, holdout_Y))
            )

            return np.array([pf, holdout_index, holdout])

        # Do these in parallel if possible
        # Pass in parallel API already constructed if possible
        if parallel_object is not None:
            parallel_results = parallel_object(
                delayed(parallel_fit)(
                    pf=penalization_factor_list[i[0]],
                    data=data,
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
                            data=data,
                            holdout_fold=folds[i[1]],
                            holdout_index=i[1],
                        )
                        for i in ff_idx
                    )
            else:
                parallel_results = Parallel(n_jobs=max_jobs)(
                    delayed(parallel_fit)(
                        pf=penalization_factor_list[i[0]],
                        data=data,
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
            cur_val = parallel_results[parallel_results[:, 0] == factor, 2].mean()

            # Calculate one standard error if requested
            if selection_type == "se":
                cur_sd = parallel_results[parallel_results[:, 0] == factor, 2].std()
                cur_se = cur_sd / np.sqrt(k_fold)
                cur_val -= cur_se

            # Check to see if the new factor value is better
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
            # self.randomize_params(**randomization_options)
            # self.optimize_params(**optimization_options)
            self.random_restart_optimize(
                data=data,
                randomize_kwargs=randomization_options,
                optimize_kwargs=optimization_options,
                num_restart=num_restart,
            )

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

    def cut_kernel_components(self, data=None, var_cutoff: float = 0.1):
        """Prune out kernel components with small variance parameters and large
        lengthscale parameters (w.r.t. input domain).

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
            self.kernel = gpflow.kernels.Sum([self.kernel.kernels[i] for i in var_flag])
        elif len(var_flag) == 1:
            if len(var_parts) > 1:
                self.kernel = self.kernel.kernels[var_flag[0]]
            else:
                self.kernel = self.kernel
        else:
            self.kernel = gpflow.kernels.Constant()

        # Prune by lengthscale as well
        if hasattr(self.kernel, "kernels"):
            self.kernel = search_through_kernel_list_(
                kernel_list=self.kernel.kernels,
                list_type=self.kernel.name,
                X=data[0].numpy(),  # self.X
            )

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
        num_latent_gps=1,
        dtype=None,
        penalized_options={},
        sparse_options={},
        variational_options={},
    ):

        super().__init__(
            X=X,
            Y=Y,
            mean_function=deepcopy(mean_function),
            kernel=deepcopy(kernel),
            verbose=verbose,
            num_latent_gps=num_latent_gps,
            dtype=dtype,
            **penalized_options,
            **sparse_options,
            **variational_options,
        )


def _describe_kernel(kern, var_names):
    """Recursively describe a kernel using variable names."""
    ktype = type(kern).__name__
    if hasattr(kern, "kernels"):
        sep = " × " if ktype == "Product" else " + "
        return sep.join(_describe_kernel(k, var_names) for k in kern.kernels)
    dims = getattr(kern, "active_dims", None)
    if dims is not None:
        dims = np.asarray(dims).flatten()
        names = [var_names[d] if d < len(var_names) else f"dim{d}" for d in dims]
        label = ", ".join(names)
    else:
        label = "all"
    short = {"SquaredExponential": "SE", "Categorical": "Cat",
             "Matern12": "Mat12", "Matern32": "Mat32", "Matern52": "Mat52",
             "Linear": "Lin", "Periodic": "Per"}.get(ktype, ktype)
    return f"{short}({label})"


class MultiOutputPSVGP(PSVGP):
    """
    Multi-output Penalized Sparse Variational Gaussian Process.
    Uses Linear Coregionalization kernel.
    """

    def __init__(
        self,
        X,
        Y,
        latent_kernels=None,
        mean_function=gpflow.functions.Constant(c=0.0),
        verbose=False,
        num_latent_gps=None,
        penalization_factor=1.0,
        dtype=None,
        # Arguments for full kernel build
        kernel_options={},
        cat_vars=[],
        num_vars=[],
        unit_idx=None,
        var_names=None,
        **kwargs,
    ):

        num_outputs = Y.shape[1]

        # Build fully saturated kernel if no latent kernels provided
        if latent_kernels is None:
            # Estimate rank if not provided in kernel_options
            if "ranks" not in kernel_options:
                # Determine likelihood for transform decision
                var_opts = kwargs.get("variational_options", {})
                lik_str = var_opts.get("likelihood", "gaussian")
                count_likelihoods = [
                    "poisson",
                    "negative_binomial",
                    "negativebinomial",
                    "zeroinflated_negativebinomial",
                ]
                transform_counts = lik_str in count_likelihoods

                # Fixed rank for continuous kernels
                estimated_rank = 3
                scale_adjustment = np.sqrt(estimated_rank)

                if verbose:
                    print(
                        f"No rank provided. Using fixed rank Q={estimated_rank} for continuous kernels."
                    )
                    print(
                        f"Penalization factor will be adjusted by sqrt(Q)={scale_adjustment:.2f}"
                    )

                # Update kernel_options (copy to avoid side effects if reused)
                kernel_options = kernel_options.copy()
                # Categoricals and unit get ranks=1 (multiple copies are unidentifiable)
                # Continuous vars get ranks=estimated_rank (can capture multiple trajectory shapes)
                ranks_dict = {idx: 1 for idx in (list(cat_vars) + ([unit_idx] if unit_idx is not None else []))}
                ranks_dict.update({idx: estimated_rank for idx in num_vars})
                kernel_options["ranks"] = ranks_dict

            from .regularization import full_kernel_build

            # Default kernel options if not provided
            default_kernel_options = {
                "second_order_numeric": False,
                "categorical_numeric_interactions": True,
                "unit_numeric_interactions": False,
                "kerns": [gpflow.kernels.SquaredExponential()],
            }
            # Merge with provided options
            k_opts = {**default_kernel_options, **kernel_options}

            # Inject num_outputs to allow default ranks to match
            k_opts["num_outputs"] = num_outputs

            # If num_vars is empty, assume all columns are numeric
            # (unless cat_vars is specified, then the rest are numeric)
            if not num_vars and not cat_vars:
                # If neither are specified, assume all are numeric
                num_vars = list(range(X.shape[1]))
            elif not num_vars:
                # If cat_vars is specified, the rest are numeric
                all_indices = set(range(X.shape[1]))
                cat_indices = set(cat_vars)
                num_vars = list(all_indices - cat_indices)

            kernel_build_result = full_kernel_build(
                cat_vars=cat_vars,
                num_vars=num_vars,
                unit_idx=unit_idx,
                var_names=var_names,
                return_sum=False,  # We want a list of kernels for LMC
                **k_opts,
            )

            # full_kernel_build returns a tuple (kernels, names) if var_names is not None
            if isinstance(kernel_build_result, tuple):
                latent_kernels = kernel_build_result[0]
            else:
                latent_kernels = kernel_build_result

            if verbose:
                print(f"Built {len(latent_kernels)} latent kernels.")

        if num_latent_gps is None:
            num_latent_gps = len(latent_kernels)

        # Initialize W to small random values
        W_init = np.random.normal(scale=0.01, size=(num_outputs, num_latent_gps))

        # Construct kernel
        kernel = gpflow.kernels.LinearCoregionalization(
            kernels=latent_kernels, W=W_init
        )

        # Add LogNormal prior to length scales of continuous kernels.
        # LogNormal(1.0, 0.5) gives median ~2.7 and 90% CI ~[1.1, 6.6]
        # in standardized input units, strongly encouraging smooth biological signals
        # while still allowing the likelihood to win if short-scale structure is real.
        ls_prior = tfd.LogNormal(loc=tf.cast(1.0, gpflow.default_float()),
                                 scale=tf.cast(0.5, gpflow.default_float()))
        for k in latent_kernels:
            if hasattr(k, "lengthscales"):
                k.lengthscales.prior = ls_prior
                k.lengthscales.assign(3.0)
            # Handle product kernels (e.g. Periodic wrapping SE)
            if hasattr(k, "kernels"):
                for sub_k in k.kernels:
                    if hasattr(sub_k, "lengthscales"):
                        sub_k.lengthscales.prior = ls_prior
                        sub_k.lengthscales.assign(3.0)

        # Manually handle sparse and variational options to ensure correct init
        sparse_options = kwargs.pop("sparse_options", {})
        variational_options = kwargs.pop("variational_options", {})

        # This is the crucial part: we are now bypassing the complex super() chain
        # and directly initializing the underlying SVGP with the correct components.

        # 1. Handle Sparse Inducing Variables
        # Reduce default inducing points to 100 for efficiency
        default_num_inducing = 100
        num_inducing_points = sparse_options.get(
            "num_inducing_points", min(X.shape[0], default_num_inducing)
        )

        if num_inducing_points >= X.shape[0]:
            # Use full data if requested points exceed data size
            inducing_variable = iv.SeparateIndependentInducingVariables(
                [iv.InducingPoints(X.copy()) for _ in range(num_latent_gps)]
            )
        else:
            # Initialize inducing points intelligently per kernel
            iv_list = []

            # Helper to get numpy array from DataFrame/Tensor
            if hasattr(X, "to_numpy"):
                X_np = X.to_numpy()
            elif tf.is_tensor(X):
                X_np = X.numpy()
            else:
                X_np = np.array(X)

            for i in range(num_latent_gps):
                k = latent_kernels[i]

                # Check for active dimensions
                if hasattr(k, "active_dims") and k.active_dims is not None:
                    dims = np.arange(X_np.shape[1])[k.active_dims]

                    # If 1D active dimension, handle based on kernel type
                    if len(dims) == 1:
                        dim_idx = dims[0]

                        # Create Z matrix initialized with means (or zeros)
                        Z = np.mean(X_np, axis=0, keepdims=True)
                        Z = np.repeat(Z, num_inducing_points, axis=0)

                        # A) Categorical: Use unique values
                        if isinstance(k, Categorical):
                            unique_vals = np.unique(X_np[:, dim_idx])

                            if len(unique_vals) >= num_inducing_points:
                                # Sample from unique values
                                np.random.seed(sparse_options.get("random_seed"))
                                z_grid = np.random.choice(
                                    unique_vals, num_inducing_points, replace=False
                                )
                            else:
                                # Tile unique values to fill num_inducing_points
                                z_grid = np.tile(
                                    unique_vals,
                                    int(
                                        np.ceil(num_inducing_points / len(unique_vals))
                                    ),
                                )
                                z_grid = z_grid[:num_inducing_points]

                        # B) Numeric: Use linspace grid
                        else:
                            # Get range of data for this dimension
                            min_val = np.min(X_np[:, dim_idx])
                            max_val = np.max(X_np[:, dim_idx])

                            # Create linspace grid
                            z_grid = np.linspace(min_val, max_val, num_inducing_points)

                        # Replace the active dimension column with selected values
                        Z[:, dim_idx] = z_grid

                        iv_list.append(iv.InducingPoints(Z))
                        continue

                # Default fallback: Random subset sampling
                np.random.seed(sparse_options.get("random_seed"))
                idx = np.random.choice(X.shape[0], num_inducing_points, replace=False)
                Z = X_np[idx, :].copy()
                iv_list.append(iv.InducingPoints(Z))

            inducing_variable = iv.SeparateIndependentInducingVariables(iv_list)

        # 2. Handle Likelihood
        likelihood_str = variational_options.get("likelihood", "gaussian")
        if likelihood_str in ("negativebinomial", "negative_binomial") and num_outputs > 1:
            # Per-output offsets and dispersion for multi-output NB
            if hasattr(Y, "to_numpy"):
                Y_np = Y.to_numpy()
            else:
                Y_np = np.array(Y)

            # Offset = log(median) per output; clip to avoid log(0)
            col_medians = np.maximum(np.median(Y_np, axis=0), 1.0)
            offsets = np.log(col_medians)

            # Dispersion via method of moments: alpha = (Var - mu) / mu^2
            col_means = np.mean(Y_np, axis=0)
            col_vars = np.var(Y_np, axis=0)
            # Clip: alpha must be positive; fallback to 1.0 where variance <= mean
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha_init = np.where(
                    col_vars > col_means,
                    (col_vars - col_means) / np.maximum(col_means ** 2, 1e-8),
                    1.0,
                )
            alpha_init = np.clip(alpha_init, 1e-4, 1e4)

            if verbose:
                print(f"NB per-output offsets: median range [{offsets.min():.1f}, {offsets.max():.1f}]")
                print(f"NB per-output alpha init: range [{alpha_init.min():.4f}, {alpha_init.max():.4f}]")

            from .likelihoods import NegativeBinomialPerOutput
            likelihood = NegativeBinomialPerOutput(alpha=alpha_init, offset=offsets)
        else:
            likelihood = gp_likelihood_crosswalk(likelihood_str)

        # 3. Call the grandparent SVGP constructor directly
        gpflow.models.SVGP.__init__(
            self,
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            num_latent_gps=num_latent_gps,
        )

        # 4. Manually set data and other attributes from our own class hierarchy
        self.data = (
            tf.convert_to_tensor(X, dtype=gpflow.default_float()),
            tf.convert_to_tensor(Y, dtype=gpflow.default_float()),
        )
        self.mean_function = mean_function
        self.verbose = verbose

        # 5. Set priors
        # Scale penalization factor by sqrt(Q) to maintain constant total prior variance
        # regardless of the number of latent components.
        scale_adjustment = np.sqrt(num_latent_gps)
        adjusted_penalization = penalization_factor * scale_adjustment

        if verbose:
            print(
                f"Horseshoe prior to W with adjusted penalization: {penalization_factor:.2f} * sqrt({num_latent_gps}) -> {adjusted_penalization:.2f}"
            )

        self.kernel.W.prior = tfd.Horseshoe(
            scale=gpflow.utilities.to_default_float(
                1.0 / adjusted_penalization if adjusted_penalization > 0 else 1.0
            )
        )

        # Freeze variance parameters of latent kernels
        from .utilities import freeze_variance_parameters

        freeze_variance_parameters(self.kernel)

    def plot_kl_scree(self, var_names=None, collapse_threshold=0.01, ax=None):
        """
        Plot a KL divergence scree plot for each latent factor.

        Bars are colored by informativeness: steelblue = informative,
        tomato = collapsed (KL < collapse_threshold * median KL).
        X-axis labels show the kernel structure for each factor.

        Args:
            var_names (list, optional): Feature names for kernel label resolution.
                Defaults to self.var_names if available.
            collapse_threshold (float): Fraction of median KL below which a
                factor is considered collapsed. Default 0.01 (1% of median).
            ax (matplotlib.axes.Axes, optional): Axes to plot into. If None,
                creates a new figure.

        Returns:
            kl_arr (np.ndarray): Per-latent KL values (unsorted, original order).
        """
        import matplotlib.pyplot as plt
        from gpflow.kullback_leiblers import gauss_kl

        q_mu_np   = self.q_mu.numpy()
        q_sqrt_np = self.q_sqrt.numpy()
        L = q_mu_np.shape[1]

        kl_arr = np.array([
            float(gauss_kl(
                tf.constant(q_mu_np[:, k:k+1]),
                tf.constant(q_sqrt_np[k:k+1]),
            ))
            for k in range(L)
        ])

        sorted_idx = np.argsort(kl_arr)[::-1]
        sorted_kl  = kl_arr[sorted_idx]
        kl_median  = np.median(kl_arr)
        collapse_mask = kl_arr < collapse_threshold * kl_median

        # Resolve kernel labels
        if var_names is None:
            var_names = getattr(self, "var_names", []) or []
        labels = []
        for k in range(L):
            kern = self.kernel.kernels[sorted_idx[k]]
            labels.append(_describe_kernel(kern, var_names))

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, L * 0.7), 4))

        colors = [
            "tomato" if collapse_mask[sorted_idx[k]] else "steelblue"
            for k in range(L)
        ]
        ax.bar(range(L), sorted_kl, color=colors)
        ax.axhline(
            collapse_threshold * kl_median,
            color="orange", linestyle="--",
            label=f"Collapse threshold ({collapse_threshold:.0%} of median = {collapse_threshold * kl_median:.1f})",
        )
        ax.set_xticks(range(L))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Latent factor (sorted by KL)")
        ax.set_ylabel("KL divergence")
        ax.set_title("KL scree plot — latent factor informativeness")
        ax.legend(fontsize=9)

        n_collapsed = collapse_mask.sum()
        if n_collapsed == 0:
            print("Smooth KL decay — all factors informative. No collapsed factors detected.")
        else:
            print(f"Collapsed factors (KL < {collapse_threshold:.0%} of median): {n_collapsed}/{L}")
            collapsed_kernels = [
                _describe_kernel(self.kernel.kernels[i], var_names)
                for i in np.where(collapse_mask)[0]
            ]
            print(f"  Collapsed: {collapsed_kernels}")

        return kl_arr

    def prune_latent_factors(
        self,
        loading_threshold=0.1,
        kl_threshold=None,
        keep_indices=None,
        optimize_after_prune=True,
        optimize_kwargs=None,
    ):
        """
        Prune latent factors based on W loading magnitude, KL divergence,
        or an explicit index list.

        Args:
            loading_threshold (float): Prune if max absolute W loading across
                all outputs is below this value. Default 0.1.
            kl_threshold (float, optional): Prune if per-latent KL is below
                this fraction of the *median* KL. Detects posterior collapse
                (near-zero KL). E.g. 0.01 = less than 1% of median.
            keep_indices (array-like, optional): Explicit list of latent column
                indices to keep. All other criteria are ignored when provided.
        """
        from gpflow.kullback_leiblers import gauss_kl

        W = self.kernel.W.numpy()
        latent_weight_importance = np.max(np.abs(W), axis=0)

        # Explicit index override — skip all threshold logic
        if keep_indices is not None:
            keep_indices = np.asarray(keep_indices)
        else:
            # Criterion 1: W loading magnitude
            to_prune = latent_weight_importance < loading_threshold

            # Criterion 3: collapse detection (KL < fraction of median)
            if kl_threshold is not None:
                q_mu = self.q_mu.numpy()
                q_sqrt_np = self.q_sqrt.numpy()
                L = q_mu.shape[1]
                kl_per_latent = np.array([
                    float(gauss_kl(
                        tf.constant(q_mu[:, k:k+1]),
                        tf.constant(q_sqrt_np[k:k+1]),
                    ))
                    for k in range(L)
                ])
                kl_cutoff = kl_threshold * np.median(kl_per_latent)
                to_prune_by_kl = kl_per_latent < kl_cutoff
                if self.verbose and to_prune_by_kl.any():
                    print(
                        f"KL collapse pruning: {to_prune_by_kl.sum()} latents "
                        f"(KL < {kl_threshold:.2%} of median "
                        f"{np.median(kl_per_latent):.2f})"
                    )
                to_prune = np.logical_or(to_prune, to_prune_by_kl)

            keep_indices = np.where(~to_prune)[0]

        if len(keep_indices) == 0:
            print(
                "Warning: All latent factors would be pruned! Keeping the one with max weight."
            )
            keep_indices = [np.argmax(latent_weight_importance)]

        if len(keep_indices) == W.shape[1]:
            if self.verbose:
                print("No latent factors pruned.")
            return

        if self.verbose:
            pruned_count = W.shape[1] - len(keep_indices)
            print(
                f"Pruning {pruned_count} latent factors. Keeping {len(keep_indices)}."
            )

        # Rebuild model with kept components
        new_kernels = [self.kernel.kernels[i] for i in keep_indices]
        new_W = W[:, keep_indices]

        # Preserve prior on W (if any) before we replace the kernel
        old_W_prior = getattr(self.kernel.W, "prior", None)
        new_q_mu = self.q_mu.numpy()[:, keep_indices]
        new_q_sqrt = self.q_sqrt.numpy()[keep_indices, :, :]

        if isinstance(self.inducing_variable, iv.SeparateIndependentInducingVariables):
            old_ivs = self.inducing_variable.inducing_variables
            new_ivs = [old_ivs[i] for i in keep_indices]
            new_inducing_variable = iv.SeparateIndependentInducingVariables(new_ivs)
        else:
            new_inducing_variable = self.inducing_variable

        # Create new Parameters with correct shapes
        self.kernel = gpflow.kernels.LinearCoregionalization(
            kernels=new_kernels, W=new_W
        )
        # Restore prior if one existed on the previous W parameter
        if old_W_prior is not None:
            try:
                self.kernel.W.prior = old_W_prior
            except Exception:
                # If assignment fails for any reason, continue without failing
                pass

        self.q_mu = gpflow.Parameter(new_q_mu)
        self.q_sqrt = gpflow.Parameter(
            new_q_sqrt, transform=gpflow.utilities.triangular()
        )
        self.inducing_variable = new_inducing_variable
        self.num_latent_gps = len(keep_indices)

        from .utilities import freeze_variance_parameters

        freeze_variance_parameters(self.kernel)
        self.update_kernel_name()

        # Optionally perform a short re-optimization after pruning to re-tune
        # remaining parameters to the reduced latent basis. This helps recover
        # any lost explanatory power from removed components.
        if optimize_after_prune:
            # Default optimization kwargs (warm-start)
            if optimize_kwargs is None:
                optimize_kwargs = {
                    "adam_learning_rate": 1e-3,
                    "nat_gradient_gamma": 0.05,
                    "num_opt_iter": 1000,
                    "constraint_weight": 0.1,
                }

            if self.verbose:
                print("Re-optimizing model after pruning latent factors...")

            try:
                # Call the model's optimize routine with warm-start settings
                self.optimize_params(
                    adam_learning_rate=optimize_kwargs.get("adam_learning_rate", 1e-3),
                    nat_gradient_gamma=optimize_kwargs.get("nat_gradient_gamma", 0.05),
                    num_opt_iter=optimize_kwargs.get("num_opt_iter", 1000),
                    constraint_weight=optimize_kwargs.get("constraint_weight", 0.1),
                )
            except Exception as e:
                if self.verbose:
                    print(f"Warning: re-optimization after pruning failed: {e}")

    def get_module_membership(
        self,
        output_names=None,
        method="otsu",
        threshold=0.2,
    ):
        """Return module membership for each latent factor.

        For each factor k, assigns outputs (e.g. metabolites) whose absolute
        loading |W_{ik}| exceeds a threshold.  Two thresholding methods are
        supported:

        - ``'otsu'``  — Otsu's method finds the threshold that minimises
          within-class variance of |W[:,k]|, exploiting the bimodal
          signal/noise distribution induced by the horseshoe prior.  Falls
          back to ``'fixed'`` when all values are identical.
        - ``'fixed'`` — threshold = ``threshold * max(|W[:,k]|)``.

        Args:
            output_names (list[str], optional): Names for the model outputs
                (e.g. metabolite IDs).  Defaults to ``["Out_0", "Out_1", ...]``.
            method (str): ``'otsu'`` (default) or ``'fixed'``.
            threshold (float): Relative threshold used when ``method='fixed'``
                or as fallback for ``'otsu'``.  Ignored for pure Otsu runs.

        Returns:
            list[tuple[str, list[str]]]: Each entry is
            ``(factor_name, [member_output_names])``, one per factor that has
            at least one member.  Factors are named ``"Factor_1"``, etc.
        """
        W = self.kernel.W.numpy()  # shape: (n_outputs, n_latents)
        n_outputs, n_latents = W.shape

        if output_names is None:
            output_names = [f"Out_{i}" for i in range(n_outputs)]

        def _otsu(values):
            vals = values.flatten()
            candidates = np.unique(vals)
            if len(candidates) < 2:
                return vals[0]
            best_t, best_var = candidates[0], np.inf
            n = len(vals)
            for t in candidates[:-1]:
                below = vals[vals <= t]
                above = vals[vals > t]
                if len(below) == 0 or len(above) == 0:
                    continue
                w0, w1 = len(below) / n, len(above) / n
                within_var = w0 * below.var() + w1 * above.var()
                if within_var < best_var:
                    best_var = within_var
                    best_t = t
            return best_t

        modules = []
        for k in range(n_latents):
            abs_w = np.abs(W[:, k])
            max_w = abs_w.max()
            if max_w < 1e-6:
                continue
            if method == "otsu" and len(np.unique(abs_w)) > 1:
                try:
                    t = _otsu(abs_w)
                except Exception:
                    t = max_w * threshold
            else:
                t = max_w * threshold
            members = [output_names[i] for i in np.where(abs_w > t)[0]]
            if members:
                modules.append((f"Factor_{k + 1}", members))

        return modules

    def optimize_params(
        self,
        adam_learning_rate=0.01,
        nat_gradient_gamma=0.1,
        num_opt_iter=2000,
        constraint_weight=1.0,
        grad_clip_norm=1e6,
        **kwargs,
    ):
        # Optimization logic from notebook

        # Optimizers
        adam_opt = tf.keras.optimizers.legacy.Adam(learning_rate=adam_learning_rate)
        natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=nat_gradient_gamma)
        natgrad_vars = [(self.q_mu, self.q_sqrt)]
        gpflow.set_trainable(self.q_mu, False)
        gpflow.set_trainable(self.q_sqrt, False)

        def loss_fn():
            return self.training_loss(self.data)

        @tf.function
        def optimization_step():
            natgrad_opt.minimize(loss_fn, var_list=natgrad_vars)

            # Optimize kernel and likelihood parameters (+ variational if no natgrad)
            with tf.GradientTape() as tape:
                loss = loss_fn()

                # Add identifiability constraints
                W = self.kernel.W

                # Weak sign constraint: encourage first element to be positive
                # W is (num_outputs, num_latents)
                sign_penalty = tf.reduce_sum(tf.nn.relu(-W[0, :]))

                total_loss = (
                    loss
                    + gpflow.utilities.to_default_float(constraint_weight)
                    * sign_penalty
                )

            grads = tape.gradient(total_loss, self.trainable_variables)

            # Global gradient clipping: preserves relative magnitudes across all params
            raw_grads = [
                g if g is not None else tf.zeros_like(v)
                for g, v in zip(grads, self.trainable_variables)
            ]
            clipped_grads, global_norm = tf.clip_by_global_norm(raw_grads, grad_clip_norm)

            adam_opt.apply_gradients(zip(clipped_grads, self.trainable_variables))
            return total_loss, loss, global_norm

        # Loop
        loss_history = []
        best_loss = float("inf")
        patience = 500
        iterations_no_improve = 0

        # Save initial values
        previous_values = gpflow.utilities.deepcopy(
            gpflow.utilities.parameter_dict(self)
        )

        for i in range(num_opt_iter):
            try:
                total_loss, data_loss, global_norm = optimization_step()
            except (tf.errors.InvalidArgumentError, tf.errors.OpError) as e:
                if self.verbose:
                    print(f"Optimization failed at step {i} with error: {e}")
                    print("Restoring previous parameter values and stopping.")
                gpflow.utilities.multiple_assign(self, previous_values)
                break

            loss_val = data_loss.numpy()

            if self.verbose and i % 500 == 0:
                W_vals = np.abs(self.kernel.W.numpy())
                w_col_max = W_vals.max(axis=0)
                n_active = np.sum(w_col_max >= 0.1)
                print(
                    f"Iteration {i}: Loss = {loss_val:.2f}, "
                    f"Active factors: {n_active}/{W_vals.shape[1]}"
                )

            # Save checkpoint
            if i % 100 == 0:
                previous_values = gpflow.utilities.deepcopy(
                    gpflow.utilities.parameter_dict(self)
                )

            # Check for NaN/Inf
            if np.isnan(loss_val) or np.isinf(loss_val):
                if self.verbose:
                    print(
                        f"Iteration {i}: WARNING - Loss became NaN/Inf: Loss = {loss_val}"
                    )
                    print("Stopping optimization to prevent divergence.")
                # Restore previous values
                gpflow.utilities.multiple_assign(self, previous_values)
                break

            # Early stopping based on loss improvement
            if loss_val < best_loss:
                best_loss = loss_val
                iterations_no_improve = 0
            else:
                iterations_no_improve += 1
                if iterations_no_improve >= patience:
                    if self.verbose:
                        print(
                            f"Iteration {i}: Early stopping - no improvement for {patience} iterations"
                        )
                    break

        self.optimizer = "custom_multioutput"


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
#             inducing_variable=gpflow.inducing_variables.InducingPoints(
#                  data[0]
#              ),
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
