import copy

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import deepcopy

# Not currently using below because it presents warning on M1 mac
# from tensorflow.keras.optimizers import Adam
# import tensorflow_probability as tfp
# from gpflow.base import RegressionData
from joblib import Parallel, delayed
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow_probability import distributions as tfd
from tqdm import tqdm

from .kernels import Empty
from .predictions import gp_predict_fun, pred_kernel_parts
from .regularization import make_folds
from .utilities import (  # hmc_sampling,
    calc_bic,
    calc_deviance_explained_components,
    calc_rsquare,
    convert_data_to_tensors,
    find_variance_components,
    find_variance_components_tf,
    gp_likelihood_crosswalk,
    print_kernel_names,
    search_through_kernel_list_,
    tqdm_joblib,
    variance_contributions,
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
        mean_function: gpflow.functions.Function = (
            gpflow.functions.Constant(c=0.0)
        ),
        kernel: gpflow.kernels.Kernel = (
            gpflow.kernels.SquaredExponential(active_dims=[0])
        ),
        verbose: bool = False,
        dtype="float64",
        **svgp_kwargs,
    ):
        # Set values
        # Comment this out for now to save memory
        # self.X = X.astype(gpflow.default_float())
        # self.Y = Y.astype(gpflow.default_float())
        # self.data = (
        #     tf.reshape(
        #         tensor=tf.convert_to_tensor(
        #             X,
        #             dtype=gpflow.default_float()
        #             # dtype=tf.float64 if dtype == "float64" else tf.float32
        #         ),
        #         shape=(X.shape)
        #     ),
        #     tf.reshape(
        #         tensor=tf.convert_to_tensor(
        #             Y,
        #             dtype=gpflow.default_float()
        #             # dtype=tf.float64 if dtype == "float64" else tf.float32
        #         ),
        #         shape=(-1, 1)
        #     )
        # )
        self.mean_function = deepcopy(mean_function)
        self.kernel = deepcopy(kernel)
        self.kernel_name = ""
        self.verbose = verbose
        self.optimizer = None
        self.num_trainable_params = np.nan

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
            mean_function=deepcopy(mean_function),
            kernel=deepcopy(kernel),
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
                        scale=scale,
                        size=self.num_inducing_points
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
                    loc=loc,
                    scale=scale,
                    size=p.numpy().shape
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
            (self.num_trainable_params <= 5000 and optimizer is None)
            or optimizer == "scipy"
        ):
            if self.verbose:
                print(
                    f"Number of params: {self.num_trainable_params},",
                    " using Scipy optimizer"
                )
                print(gpflow.utilities.print_summary(self))
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
                        options=opt_options
                    )
                    break
                except Exception as e:
                    if self.verbose:
                        print(f"Attempt {attempt+1} - Scipy exception: {e}")
                    for param in self.parameters:
                        param = tf.cast(param, tf.float64)

            return None
        
        if (
            (self.num_trainable_params > 5000 and optimizer is None)
            or optimizer == "adam/gradient"
        ):
            # Set optimizer otherwise
            self.optimizer = "adam/gradient"

            # Stop Adam from optimizing the variational parameters
            gpflow.set_trainable(self.q_mu, False)
            gpflow.set_trainable(self.q_sqrt, False)

            # Create the optimize_tensors for VGP with natural gradients
            adam_opt = Adam(learning_rate=adam_learning_rate)
            natgrad_opt = gpflow.optimizers.NaturalGradient(
                gamma=nat_gradient_gamma
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
                " Current options: ['scipy', 'adam', 'adam/gradient', None]"
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
            compiled_loss = self.training_loss_closure(
                iter(data_minibatch)
            )
        else:
            # Compile loss closure based on training data
            compiled_loss = self.training_loss_closure(
                data=data,
                compile=True
            )

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
                    "Reached invalid step in optimization,"
                    " returning previous step."
                )
                gpflow.utilities.multiple_assign(self, previous_values)
                break

            # Checkpoint!
            if i % 100 == 0:
            # if i % 500 == 0:
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
                    adam_opt.learning_rate = (
                        adam_learning_rate *
                        adam_decay_rate ** (i / 500)
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
                        print(
                            f"Optimization converged - stopping early (round {i})"
                        )
                    break
        
        # If we have reached the end of iterations and still
        # not converged then...
        if i == (num_opt_iter - 1):
            if self.verbose:
                print(f"Optimization not converged after {i+1} rounds")

        return None

    def random_restart_optimize(
        self,
        data=None,
        num_restart=5,
        randomize_kwargs={},
        optimize_kwargs={}
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
            cur_max_ll = self.maximum_log_likelihood_objective(
                data=data
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

    def get_variance_explained(self, data=None):
        """Calculates variance explained for each additive kernel component.

        Arguments
        ---------
        None

        Returns
        -------
        None
            Variance contributions in self.variance_explained
        """
        # self.variance_explained = list(
        #     variance_contributions(
        #         self, k_names=self.kernel_name, lik=self.likelihood.name
        #     )
        # )
        # var_list = calc_rsquare(self, data=data)
        var_list = calc_deviance_explained_components(
            model=self,
            data=data
        )
        

        # Fix ListWrapper issue with Tensorflow tensors
        self.variance_explained = list(var_list)

        # return self.variance_explained
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
            **kwargs
        )

    def plot_parts(self, x_idx, col_names, data=None, lik=None, **kwargs):
        if lik is None:
            lik = self.likelihood
        return pred_kernel_parts(
            self,
            x_idx=x_idx,
            # unit_idx=unit_idx,
            col_names=col_names,
            var_explained=self.variance_explained,
            lik=lik,
            data=data,
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
        scale_value=None,
        # variational_priors=True,
        verbose=False,
        **basegp_kwargs,
    ):
        # Set values
        # self.X = X
        # self.Y = Y
        self.mean_function = deepcopy(mean_function)
        self.kernel = deepcopy(kernel)
        self.verbose = verbose

        # Fill in information for parent class
        super().__init__(
            X=X,
            Y=Y,
            mean_function=deepcopy(mean_function),
            kernel=deepcopy(kernel),
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
        # Set values
        # self.X = X
        # self.Y = Y
        self.mean_function = deepcopy(mean_function)
        self.kernel = deepcopy(kernel)
        self.verbose = verbose
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
            X.numpy()[obs_idx, :].copy()
            if tf.is_tensor(X)
            else X[obs_idx, :].copy()
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
            mean_function=deepcopy(mean_function),
            kernel=deepcopy(kernel),
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
                self.penalization_factor *
                find_variance_components_tf(self.kernel)
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
                    scale=gpflow.utilities.to_default_float(
                        1. / penalization_factor
                    )
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
        selection_type="se"
    ):
        
        # Check data types
        X = data[0] if isinstance(data[0], np.ndarray) else data[0].numpy()
        Y = data[1] if isinstance(data[1], np.ndarray) else data[1].numpy()
        
        # Split training data into k-folds
        folds = make_folds(
            X,
            self.unit_col,
            k_fold,
            random_seed
        )

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
        def parallel_fit(pf, data, holdout_fold, holdout_index):
            temp_model = copy.deepcopy(self)
            training_data = convert_data_to_tensors(
                X=np.delete(
                    X,
                    holdout_fold,
                    axis=0
                ),
                Y=np.delete(
                    Y,
                    holdout_fold,
                    axis=0
                )
            )
            holdout_X = X[holdout_fold]
            holdout_Y = Y[holdout_fold]
            temp_model.random_restart_optimize(
                data=training_data,
                randomize_kwargs=randomization_options,
                optimize_kwargs=optimization_options,
                num_restart=num_restart
            )
            holdout = np.mean(
                temp_model.predict_log_density(
                    data=(holdout_X, holdout_Y)
                )
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
            cur_val = parallel_results[
                parallel_results[:, 0] == factor, 2
            ].mean()

            # Calculate one standard error if requested
            if selection_type == "se":
                cur_sd = parallel_results[
                    parallel_results[:, 0] == factor, 2
                ].std()
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
                num_restart=num_restart
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
            self.kernel = gpflow.kernels.Sum(
                [self.kernel.kernels[i] for i in var_flag]
            )
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


class PSVGP(
    PenalizedGP,
    SparseGP,
    VarGP
):
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
        dtype="float64",
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
            dtype=dtype,
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
