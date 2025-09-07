import os
import re
import time
import warnings

import gpflow
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import psutil
import ray
import scipy
import seaborn as sns
import tensorflow as tf
from gpflow.utilities import set_trainable
from joblib import Parallel, delayed
from ray.experimental import tqdm_ray

# from tensorflow_probability import distributions as tfd
from tqdm import tqdm

from .kernels import Categorical, Lin

# from .likelihoods import ZeroInflatedNegativeBinomial
from .model_classes import PSVGP
from .predictions import gp_predict_fun, pred_kernel_parts
from .regularization import full_kernel_build
from .utilities import (
    ParallelTqdm,
    calc_bic,
    calc_deviance_explained_components,
    calc_rsquare,
    check_if_model_exists,
    convert_data_to_tensors,
    find_variance_components,
    print_kernel_names,
    replace_kernel_variables,
    run_ray_process,
    tqdm_joblib,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["RAY_FUNCTION_SIZE_ERROR_THRESHOLD"] = "30000000"
f64 = gpflow.utilities.to_default_float


class GPSearch:
    """Gaussian process model search class.

    Parameters
    ----------
    X : pandas.DataFrame
        Design dataframe including covariates of interest to make up kernel

    Y : pandas.DataFrame
        Output dataframe, each column is one output vector

    unit_col : str
        Unit column name in X

    categorical_vars : list of str
        List of column names in X that represent categorical variables

    outcome_likelihood: str
        Likelihood distribution of Y variables. Only accepts one string and
        currently only supports 'gaussian', 'bernoulli',
        or 'poisson' (experimental)

    Attributes
    ----------
    """

    def __init__(
        self,
        X,
        Y,
        unit_col=None,
        standardize_X=True,
        Y_transform=None,
        categorical_vars=[],
        outcome_likelihood="gaussian",
    ):
        X = X.copy()

        # Check input types
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a Pandas DataFrame")
        if not isinstance(Y, pd.DataFrame):
            raise TypeError("Y is not a Pandas DataFrame")

        # Add unit col to categorical variables if it isn't specified
        if unit_col is not None and unit_col not in categorical_vars:
            categorical_vars += [unit_col]

        # Transform any categorical columns to integers
        self.categorical_dict = {}

        for c in categorical_vars:
            if X[c].dtype in ["object", "string"]:
                print(f"Converting {c} to numeric")
                factor_out = pd.factorize(X[c])
                self.categorical_dict[c] = factor_out
                X.loc[:, c] = factor_out[0]
                X = X.astype({c: float})

        # Make sure all resulting columns are the same type
        float_match = X.columns.isin(X.select_dtypes(include=[float]).columns)
        if sum(float_match) != len(X.columns):
            try:
                X = X.astype(float)
            except:
                raise TypeError(
                    "X columns must all be float type."
                    f" Cast {X.columns[~float_match]} to float."
                    " Perhaps use pandas.factorize() and"
                    " pandas.DataFrame.astype()."
                )
        float_match = Y.columns.isin(Y.select_dtypes(include=[float]).columns)
        if sum(float_match) != len(Y.columns):
            try:
                Y = Y.astype(float)
            except:
                raise TypeError(
                    "Y columns must all be float type."
                    f" Cast {Y.columns[~float_match]} to float."
                    " Perhaps use pandas.factorize() and"
                    " pandas.DataFrame.astype()."
                )

        # Make sure there is no missing data
        assert (
            X.isna().sum().sum() == 0
        ), "NAs in X, waveome cannot currently handle missing values!"
        assert (
            Y.isna().sum().sum() == 0
        ), "NAs in Y, waveome cannot currently handle missing values!"

        # Load up attributes
        self.X = X.copy()
        self.Y = Y.copy()
        self.feat_names = X.columns.tolist()
        self.out_names = Y.columns.tolist()
        self.cat_idx = [self.feat_names.index(x) for x in categorical_vars]
        if unit_col is not None:
            self.unit_idx = self.feat_names.index(unit_col)
        else:
            self.unit_idx = None
        self.likelihood = outcome_likelihood

        # Pull off continuous column indexes
        self.cont_idx = np.where(
            ~np.in1d(np.arange(X.shape[1]), self.cat_idx)
        )[0].tolist()

        # Standardize continuous X columns
        if standardize_X:
            self.X_means = self.X.iloc[:, self.cont_idx].mean(axis=0)
            self.X_stds = self.X.iloc[:, self.cont_idx].std(axis=0)
            self.X_original = self.X.copy()
            # self.X.iloc[:, np.array(self.cont_idx, dtype="float")] = (
            #     (
            #         self.X.iloc[:, self.cont_idx] - self.X_means
            #     ) / self.X_stds
            # ).astype(float)
            for c in self.cont_idx:
                cont_feat_name = self.feat_names[c]
                self.X[cont_feat_name] = (
                    self.X[cont_feat_name] - self.X_means[cont_feat_name]
                ) / self.X_stds[cont_feat_name]

        # Transform Y columns
        # It might only be useful for some likelihoods
        if Y_transform == "standardize":
            if self.likelihood != "gaussian":
                warnings.warn(
                    "Standardizing Y without a gaussian likelihood is"
                    " not advised! Maybe Y_transform='scale' is better?"
                )
            self.Y_means = self.Y.mean(axis=0)
            self.Y_stds = self.Y.std(axis=0)
            self.Y_original = self.Y.copy()
            self.Y = (self.Y - self.Y_means) / self.Y_stds
        elif Y_transform == "scale":
            if self.likelihood in ["binomial", "bernoulli"]:
                warnings.warn(
                    f"Scaling Y with {outcome_likelihood} is"
                    " not advised! Maybe pass as-is with Y_transform=None"
                    " is better?"
                )
            self.Y_stds = self.Y.std(axis=0)
            self.Y_original = self.Y.copy()
            self.Y = self.Y / self.Y_stds
        
        # Unclear if we still need this step
        # elif Y_transform is None and self.likelihood == "negativebinomial":
        #     self.Y_stds = self.Y.std(axis=0)

    def penalized_optimization(
        self,
        full_kernel=None,
        num_jobs=-1,
        verbose=False,
        mean_function=gpflow.mean_functions.Constant(),
        kernel_options={
            "second_order_numeric": False,
            "unit_numeric_interactions": False,
            "kerns": [
                gpflow.kernels.SquaredExponential(),
                # gpflow.kernels.Matern12(),
                # Lin(),
                # gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
            ],
        },
        penalization_factor=1.0,
        num_factor_iter=5,
        num_restart=0,
        sparse_options={},
        variational_options={},
        optimization_options={
            # "optimizer": "adam/gradient",
            # "minibatch_size": 64
            "optimizer": "scipy"
        },
        random_seed=None,
        ray_dashboard=False,
        ray_logging=False
    ):
        # Set model selection type
        self.model_selection_type = "penalized"

        # Set seed if requested
        if random_seed is not None:
            np.random.seed(random_seed)

        # If kernel is none then build out full kernel set
        if full_kernel is None:
            full_kernel, full_kernel_name = full_kernel_build(
                cat_vars=self.cat_idx,
                num_vars=self.cont_idx,
                unit_idx=self.unit_idx,
                var_names=self.feat_names,
                return_sum=True,
                **kernel_options,
            )
        else:
            full_kernel = gpflow.utilities.deepcopy(full_kernel)

        # Add likelihood information
        variational_options["likelihood"] = self.likelihood

        # Parallel model building function
        @ray.remote(max_calls=1, max_retries=5)
        def model_build_steps_remote(
                self_X,
                self_Y,
                self_likelihood,
                self_Y_stds,
                self_penalization_factor,
                self_mean_function,
                self_full_kernel,
                feat,
                tqdm_bar
        ):

            # # Add scale if negative binomial
            # if (self_likelihood == "negativebinomial" and
            #         self_Y_stds is not None):
            #     variational_options["scale_value"] = self_Y_stds[feat]

            # Set penalization factor if None passed in
            if self_penalization_factor is None:
                num_params = len(
                    find_variance_components(self_full_kernel, sum_reduce=False)
                )

                # If we don't want to iterate sigma we set to one
                if num_factor_iter == 0:
                    sigma_hat = 1
                else:
                    sigma_hat = np.std(self_Y[feat])

                self_penalization_factor = (
                    2
                    * 1.1
                    * sigma_hat
                    * np.sqrt(self_X.shape[0])
                    * scipy.stats.norm().ppf(1-(0.1 / (2 * num_params)))
                )

                self.iterating_penalization_factor = True

                if verbose:
                    print(
                        "Setting penalization factor to",
                        f" {self_penalization_factor}"
                    )
            else:
                self.iterating_penalization_factor = False

            # Specify model
            mod = PSVGP(
                X=self_X.to_numpy(),
                Y=self_Y[feat].to_numpy().reshape(-1, 1),
                mean_function=gpflow.utilities.deepcopy(self_mean_function),
                kernel=gpflow.utilities.deepcopy(self_full_kernel),
                verbose=verbose,
                penalized_options={
                    "penalization_factor": self_penalization_factor
                },
                sparse_options=sparse_options,
                variational_options=variational_options,
            )
            
            # Random restarts to find optimal parameters
            if num_restart > 0:
                mod.random_restart_optimize(
                    data=convert_data_to_tensors(
                        self_X.to_numpy(),
                        self_Y[feat].to_numpy().reshape(-1, 1)
                    ),
                    num_restart=num_restart,
                    randomize_kwargs={"random_seed": random_seed},
                    optimize_kwargs=optimization_options,
                )
            else:
                mod.optimize_params(
                    data=convert_data_to_tensors(
                        self_X.to_numpy(),
                        self_Y[feat].to_numpy().reshape(-1, 1)
                    ),
                    **optimization_options,
                )

            # If no penalization factor set then we iterater for number of steps
            if self.iterating_penalization_factor is True:

                for _ in np.arange(num_factor_iter):

                    # Store previous parameter dictionary
                    prev_params = gpflow.utilities.read_values(self)

                    # Estimate residual standard deviation
                    new_sd = np.sqrt(np.mean(mod.predict_y(self_X.values)[1]))

                    new_penalization_factor = (
                        2
                        * 1.1
                        * new_sd
                        * np.sqrt(self_X.shape[0])
                        * scipy.stats.norm().ppf(1-(0.1 / (2 * num_params)))
                    )

                    if verbose:
                        print(
                            "New penalization factor:"
                            f" {new_penalization_factor}"
                        )

                    # Break out if the new factor is similar to current factor
                    if (
                        abs(new_penalization_factor - mod.penalization_factor)
                        <= 1e-3
                    ):
                        break

                    # Assign previous values and break out if new factor is larger
                    if new_penalization_factor > mod.penalization_factor:
                        if verbose:
                            print("Larger penalization factor, assigning previous values and exiting")
                        gpflow.utilities.multiple_assign(self, prev_params)
                        break

                    # Break out if the new factor is larger than the current factor
                    mod.set_penalization_factor(new_penalization_factor)

                    # Continue to optimize current parameters
                    mod.optimize_params(
                        **optimization_options,
                        data=convert_data_to_tensors(
                            self_X.to_numpy(),
                            self_Y[feat].to_numpy().reshape(-1, 1)
                        )
                    )

            # Clean up final model
            mod.cut_kernel_components(
                data=convert_data_to_tensors(self_X, self_Y)
            )

            # TODO: Prune kernel (specifically interactions)

            mod.update_kernel_name()
            mod.get_variance_explained(
                data=convert_data_to_tensors(
                    self_X.to_numpy(),
                    self_Y[feat].to_numpy().reshape(-1, 1)
                )
            )

            # Increment progress bar (add time for response)
            tqdm_bar.update.remote(1)
            time.sleep(10)

            return mod

        # # Joblib parallel for model building
        # out = ParallelTqdm(
        #     desc="Penalized optimization (no search)",
        #     total_tasks=len(self.out_names),
        #     n_jobs=num_jobs,
        #     backend="loky"
        # )([
        #     delayed(model_build_steps)(
        #         self.X,
        #         self.Y,
        #         self.likelihood,
        #         self.Y_stds,
        #         feat
        #     ) for feat in self.out_names
        # ])
        
        # Loop to build models on Ray cluster
        self.models = {}

        # Make sublists of outcomes
        if num_jobs == -1:
            num_processes = None
            num_feats_per_round = 1000 * psutil.cpu_count()
        else:
            num_processes = num_jobs
            num_feats_per_round = 1000 * num_processes

        grouped_feat_list = [
            self.out_names[x:x+num_feats_per_round]
            for x in range(0, len(self.out_names), num_feats_per_round)
        ]

        # Set up ray tqdm tracker
        remote_tqdm = ray.remote(tqdm_ray.tqdm)

        print(f"Building {len(self.out_names)} models...")
        start_time = time.time()
        c = 0
        num_feats = len(self.out_names)
        for i in grouped_feat_list:
            # Initialize ray
            try:
                ray.init(
                    num_cpus=num_processes,
                    include_dashboard=ray_dashboard,
                    configure_logging=ray_logging
                )
            except RuntimeError:
                ray.shutdown()
                ray.init(
                    num_cpus=num_processes,
                    include_dashboard=ray_dashboard,
                    configure_logging=ray_logging
                )


            # Put main information in shared data store
            # self_ref = ray.put(self)
            self_X = ray.put(self.X)
            self_Y = ray.put(self.Y)
            self_likelihood = ray.put(self.likelihood)
            if hasattr(self, "Y_stds"):
                self_Y_stds = ray.put(self.Y_stds)
            else:
                self_Y_stds = ray.put(None)
            self_penalization_factor = ray.put(penalization_factor)
            self_mean_function = ray.put(mean_function)
            self_full_kernel = ray.put(full_kernel)

            # # Load function
            # model_build_steps_remote = ray.remote(
            #     model_build_steps,
            #     max_calls=1,
            #     max_retries=5
            # )

            # Create progress bar
            bar = remote_tqdm.remote(total=len(i))

            # Retrieve models
            out = ray.get([
                model_build_steps_remote.remote(
                    self_X,
                    self_Y,
                    self_likelihood,
                    self_Y_stds,
                    self_penalization_factor,
                    self_mean_function,
                    self_full_kernel,
                    feat,
                    bar
                )
                for feat in i  # self.out_names
            ])

            # Add models to dictionary
            for m, feat in zip(out, i):
                self.models[feat] = m

            # Close progress bar
            bar.close.remote()

            # Shut down ray and remove temporary directory
            ray.shutdown()
            os.system("rm -rf /tmp/ray")

            # Add number of finished models
            c += len(i)

            # Print output
            # if c % 10 == 0 and c > 0:
            prop_done = int(np.round(100*c/num_feats))
            elapsed_time = np.round((time.time() - start_time)/60, 1)
            print(
                f"Finished {c} models ({prop_done}%),",
                f"elapsed time: {elapsed_time} minutes"
            )

        # # Load up model dictionary
        # self.models = {feat: mod for feat, mod in zip(self.out_names, out)}

        return None

    # def run_penalized_search(
    #     self,
    #     full_kernel=None,
    #     num_jobs=-2,
    #     verbose=False,
    #     mean_function=gpflow.functions.Constant(c=0.0),
    #     kernel_options={
    #         "second_order_numeric": False,
    #         "kerns": [
    #             gpflow.kernels.SquaredExponential(),
    #             gpflow.kernels.Matern12(),
    #             gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()),
    #             Lin()
    #         ],
    #     },
    #     penalized_options={},
    #     search_options={
    #         "num_restart": 1
    #     },
    #     sparse_options={},
    #     variational_options={},
    #     optimization_options={},
    #     random_seed=None,
    # ):
    #     # Set model selection type
    #     self.model_selection_type = "penalized"

    #     # Set seed if requested
    #     if random_seed is not None:
    #         np.random.seed(random_seed)

    #         # Also add it to the search options if not requested
    #         if "random_seed" not in search_options.keys():
    #             search_options["random_seed"] = random_seed

    #     # If kernel is none then build out full kernel set
    #     if full_kernel is None:
    #         full_kernel, full_kernel_name = full_kernel_build(
    #             cat_vars=self.cat_idx,
    #             num_vars=self.cont_idx,
    #             var_names=self.feat_names,
    #             return_sum=True,
    #             **kernel_options,
    #         )
    #     else:
    #         full_kernel = gpflow.utilities.deepcopy(full_kernel)

    #     # Add likelihood information
    #     variational_options["likelihood"] = self.likelihood

    #     # Load up model dictionary
    #     self.models = {}
    #     for feat in self.out_names:
    #         self.models[feat] = PSVGP(
    #             X=self.X.to_numpy(),
    #             Y=self.Y[feat].to_numpy().reshape(-1, 1),
    #             mean_function=gpflow.utilities.deepcopy(mean_function),
    #             kernel=gpflow.utilities.deepcopy(full_kernel),
    #             verbose=verbose,
    #             penalized_options=penalized_options,
    #             sparse_options=sparse_options,
    #             variational_options=variational_options,
    #         )

    #     # Calculate total number of tasks needed to search over
    #     k_fold = (
    #         3
    #         if "k_fold" not in search_options.keys()
    #         else search_options["k_fold"]
    #     )
    #     n_factors = (
    #         5
    #         if "penalization_factor_list" not in search_options.keys()
    #         else len(search_options["penalization_factor_list"])
    #     )
    #     tot_tasks = len(self.out_names) * k_fold * n_factors

    #     with tqdm_joblib(tqdm(desc="GPPenalized", total=tot_tasks)) as _:
    #         with Parallel(n_jobs=num_jobs) as parallel:
    #             for outcome, model in self.models.items():
    #                 model.penalization_search(
    #                     data=convert_data_to_tensors(
    #                         self.X.to_numpy(),
    #                         self.Y[outcome].to_numpy().reshape(-1, 1)
    #                     ),
    #                     parallel_object=parallel,
    #                     optimization_options=optimization_options,
    #                     **search_options,
    #                 )

    #                 # Clean up models (prune)
    #                 model.cut_kernel_components(
    #                     data=convert_data_to_tensors(
    #                         self.X.to_numpy(),
    #                         self.Y[outcome].to_numpy().reshape(-1, 1)
    #                     )
    #                 )

    #                 # Update kernel names
    #                 model.update_kernel_name()

    #                 # Also attach variance explained
    #                 model.get_variance_explained(
    #                     data=convert_data_to_tensors(
    #                         self.X.to_numpy(),
    #                         self.Y[outcome].to_numpy().reshape(-1, 1)
    #                     )
    #                 )

    #     return None

    def run_penalized_search(
        self,
        full_kernel=None,
        num_jobs=-2,
        verbose=False,
        mean_function=gpflow.functions.Constant(c=0.0),
        kernel_options={
            "second_order_numeric": False,
            "kerns": [
                gpflow.kernels.SquaredExponential(),
                gpflow.kernels.Matern12(),
                gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()),
                Lin()
            ],
        },
        penalized_options={},
        search_options={
            "num_restart": 1
        },
        sparse_options={},
        variational_options={},
        optimization_options={},
        random_seed=None,
        include_dashboard=False
    ):

        raise NotImplementedError(
            "run_penalized_search is deprecated, use penalized_optimization instead."
        )

        # Set model selection type
        self.model_selection_type = "penalized"

        # Set seed if requested
        if random_seed is not None:
            np.random.seed(random_seed)

            # Also add it to the search options if not requested
            if "random_seed" not in search_options.keys():
                search_options["random_seed"] = random_seed

        # If kernel is none then build out full kernel set
        if full_kernel is None:
            full_kernel, full_kernel_name = full_kernel_build(
                cat_vars=self.cat_idx,
                num_vars=self.cont_idx,
                unit_idx=self.unit_idx,
                var_names=self.feat_names,
                return_sum=True,
                **kernel_options,
            )
        else:
            full_kernel = gpflow.utilities.deepcopy(full_kernel)

        # Add likelihood information
        variational_options["likelihood"] = self.likelihood

        def build_models(
            X,
            Y,
            mean_function,
            full_kernel,
            search_options,
            variational_options,
            feat,
            bar
        ):

            # Instatiate new model with specifications
            model = PSVGP(
                X=X.to_numpy(),
                Y=Y[feat].to_numpy().reshape(-1, 1),
                mean_function=gpflow.utilities.deepcopy(mean_function),
                kernel=gpflow.utilities.deepcopy(full_kernel),
                verbose=verbose,
                penalized_options=penalized_options,
                sparse_options=sparse_options,
                variational_options=variational_options,
            )

            # Calculate total number of tasks needed to search over
            k_fold = (
                3
                if "k_fold" not in search_options.keys()
                else search_options["k_fold"]
            )

            # Serial processing for each ray task (model)
            model.penalization_search(
                data=convert_data_to_tensors(
                    self.X.to_numpy(),
                    self.Y[feat].to_numpy().reshape(-1, 1)
                ),
                parallel_object=None,
                max_jobs=1,
                optimization_options=optimization_options,
                k_fold=k_fold,
                show_progress=False,
                **search_options,
            )

            # Clean up models (prune)
            model.cut_kernel_components(
                data=convert_data_to_tensors(
                    self.X.to_numpy(),
                    self.Y[feat].to_numpy().reshape(-1, 1)
                )
            )

            # Update kernel names
            model.update_kernel_name()

            # Also attach variance explained
            model.get_variance_explained(
                data=convert_data_to_tensors(
                    self.X.to_numpy(),
                    self.Y[feat].to_numpy().reshape(-1, 1)
                )
            )

            # Update progress bar
            bar.update.remote(1)

            return model

        # Now run each of these tasks in parallel with Ray
        self.models = run_ray_process(
            num_jobs=num_jobs,
            model_output_names=self.out_names,
            func=build_models,
            stored_func_args={
                "X": self.X,
                "Y": self.Y,
                "mean_function": mean_function,
                "full_kernel": full_kernel,
                "search_options": search_options,
                "variational_options": variational_options,
            },
            include_ray_dashboard=include_dashboard
        )

        return None

    def run_search(
        self,
        kernels=[
            gpflow.kernels.SquaredExponential(),
            gpflow.kernels.Matern12(),
            Lin(),
            gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
        ],
        max_depth=5,
        early_stopping=True,
        prune=True,
        keep_all=False,
        metric_diff=6,
        num_restart=1,
        random_seed=None,
        num_jobs=-1,
        verbose=False,
        debug=False,
    ):
        """Run search process given search operator and kernels of interest.

        Parameters
        ----------

        Attributes
        ----------

        """

        # Set model selection type
        self.model_selection_type = "stepwise"
        self.verbose = verbose

        # # Calculate number of output columns
        # num_out = len(self.out_names)

        # # Take min of output or requested jobs for num_jobs
        # if num_jobs <= 0:
        #     num_jobs = os.cpu_count() + num_jobs + 1
        # num_jobs = min(num_out, num_jobs)

        # with tqdm_joblib(tqdm(desc="Kernel search", total=num_out)) as _:
        #     models_out = Parallel(n_jobs=num_jobs, verbose=1)(
        #         delayed(full_kernel_search)(
        #             X=self.X,
        #             Y=self.Y[self.out_names[i]],
        #             kern_list=kernels,
        #             cat_vars=self.cat_idx,
        #             max_depth=max_depth,
        #             early_stopping=early_stopping,
        #             prune=prune,
        #             keep_all=keep_all,
        #             lik=self.likelihood,
        #             metric_diff=metric_diff,
        #             num_restart=num_restart,
        #             random_seed=random_seed,
        #             verbose=verbose,
        #             debug=debug,
        #         )
        #         for i in range(num_out)
        #     )

        # # Make dictionary of outcomes as lookups
        # dict_out = {feat: mod for feat, mod in zip(self.out_names, models_out)}

        # self.search_info = dict_out
        # self.models = {
        #     feat: mod["models"][mod["best_model"]]["model"]
        #     for feat, mod in zip(self.out_names, models_out)
        # }

        # # Also calculate variance explained
        # for o, m in self.models.items():
        #     m.get_variance_explained(
        #         data=convert_data_to_tensors(
        #             self.X.to_numpy(),
        #             self.Y[o].to_numpy().reshape(-1, 1)
        #         )
        #     )

        # Updated process with Ray cluster
        self.models = {}

        # Make sublists of outcomes
        if num_jobs == -1:
            num_processes = None
            num_feats_per_round = 5 * psutil.cpu_count()
        else:
            num_processes = num_jobs
            num_feats_per_round = 5 * num_processes
        grouped_feat_list = [
            self.out_names[x:x+num_feats_per_round]
            for x in range(0, len(self.out_names), num_feats_per_round)
        ]

        print(f"Building {len(self.out_names)} models...")
        start_time = time.time()
        c = 0
        num_feats = len(self.out_names)

        # Loop through groups of outputs
        for i in grouped_feat_list:
            # Initialize ray
            try:
                ray.init(
                    num_cpus=num_processes,
                    include_dashboard=False,
                    configure_logging=False
                )
            except RuntimeError:
                ray.shutdown()
                ray.init(
                    num_cpus=num_processes,
                    include_dashboard=False,
                    configure_logging=False
                )

            # Put main information in shared data store
            self_X = ray.put(self.X)
            self_Y = ray.put(self.Y)
            self_cat_vars = ray.put(self.cat_idx)
            self_likelihood = ray.put(self.likelihood)
            if hasattr(self, "Y_stds"):
                self_Y_stds = ray.put(self.Y_stds)
            else:
                self_Y_stds = ray.put(None)

            # Load function
            full_kernel_search_remote = ray.remote(full_kernel_search)

            # Retrieve models
            out = ray.get([
                full_kernel_search_remote.remote(
                    X=self_X,
                    Y=self_Y,
                    kern_list=kernels,
                    cat_vars=self_cat_vars,
                    max_depth=max_depth,
                    early_stopping=early_stopping,
                    prune=prune,
                    keep_all=keep_all,
                    lik=self_likelihood,
                    scale_value=self_Y_stds,
                    metric_diff=metric_diff,
                    num_restart=num_restart,
                    random_seed=random_seed,
                    verbose=verbose,
                    debug=debug,
                    feature_name=feat
                )
                for feat in self.out_names
            ])

            # TODO: Make dictionary of outcomes as lookups
            # self.search_info = {feat: mod for feat, mod in zip(self.out_names, out)}

            # Add models to dictionary and calculate explained values
            for m, feat in zip(out, i):
                self.models[feat] = m["models"][m["best_model"]]["model"]
                self.models[feat].get_variance_explained(
                    data=convert_data_to_tensors(
                        self.X.to_numpy(),
                        self.Y[feat].to_numpy().reshape(-1, 1)
                    )
                )

            ray.shutdown()

            # Add number of finished models
            c += len(i)

            # Print output
            # if c % 10 == 0 and c > 0:
            prop_done = int(np.round(100*c/num_feats))
            elapsed_time = np.round((time.time() - start_time)/60, 1)
            print(
                f"Finished {c} models ({prop_done}%),",
                f"elapsed time: {elapsed_time} minutes"
            )

        return None

    # TODO: Add this functionality to create variance explained barchart
    # of outcomes
    # def plot_variance_explained(self, var_cutoff=0.8, feat=None):
    #
    #     # Subset to models that have feats of interest
    #
    #     # Get percent of variance explained for each component
    #
    #     # Normalize to get the percentage of variance explained
    #
    #     # Plot
    #     f, ax = plt.subplots(figsize=(6, 15))
    #     bp = sns.barplot(
    #         x=[var_explained[x] for x in np.argsort(var_explained)[::-1]],
    #         y=missing_mbx_list[:n_met][np.argsort(var_explained)[::-1]],
    #         color='black',
    #     )
    #     return bp

    def plot_heatmap(
        self,
        var_cutoff=0.8,
        metric_cutoff=None,
        feature_name=None,
        show_vals=True,
        figsize=None,
        cluster=True,
        print_drop_count=False,
        **clustermap_kwargs
    ):

        # Specify output dataframe
        out_info = pd.DataFrame()

        # Drop counters
        n_feature_drops = 0
        n_explained_drops = 0

        # Loop through all models
        for o in self.out_names:

            # Copy model
            m_copy = gpflow.utilities.deepcopy(self.models[o])
            var_explained = m_copy.variance_explained

            # First see if a feature is in each model
            if feature_name is not None:
                
                # Get index of specified feature
                feature_index = np.where(self.X.columns == feature_name)[0][0]
                # Figure out where it is in the model kernel
                feature_kernel_flags = [
                    str(feature_index) in y for y in [
                        re.findall(r"\[(\d+)\]", x)
                        for x in m_copy.kernel_name.split("+")
                        ]
                    ]
                
                # If this feature is in the selected model then subset model
                if sum(feature_kernel_flags) > 0 and m_copy.kernel.name == "sum":
                    feature_kernel_index = list(
                        np.where(feature_kernel_flags)[0]
                    )

                    if len(feature_kernel_index) > 1:
                        new_k = gpflow.kernels.Sum([
                            m_copy.kernel.kernels[x]
                            for x in feature_kernel_index
                        ])
                    else:
                        new_k = m_copy.kernel.kernels[feature_kernel_index[0]]
                    
                    # Grab the specific explained component + leftovers
                    var_explained = [
                        m_copy.variance_explained[x]
                        for x in (
                            feature_kernel_index
                            + [-1]
                        )
                    ]

                    m_copy.kernel = new_k
                    m_copy.update_kernel_name()

                # If this feature doesn't exist in the model then pass
                elif sum(feature_kernel_flags) == 0:
                    n_feature_drops += 1
                    continue
                    
            else:
                feature_index = None

            # # Calculate the variance explained for each component and overall
            # if (
            #     feature_index is None
            #     and self.models[o].variance_explained is not None
            # ):
            #     print("Loading previous variance explained")
            #     var_explained = self.models[o].variance_explained
            # else:
            #     var_explained = calc_deviance_explained_components(
            #         model=m_copy,
            #         data=(
            #             self.X.to_numpy(),
            #             self.Y[o].to_numpy().reshape(-1, 1)
            #         )
            #     )

            # Now check to make sure we have explained "enough" variance
            if (1 - var_explained[-1]) < var_cutoff:
                n_explained_drops += 1
                continue

            # Check if metric cutoff is specified and met
            if metric_cutoff is not None:
                if max(var_explained[:-1]) < metric_cutoff:
                    n_explained_drops += 1
                    continue

            # If we are still investigating this feature then save output for ploting
            kname = replace_kernel_variables(
                m_copy.kernel_name,
                self.feat_names
            )

            # Now add this row to our output dataframe
            new_row = pd.DataFrame(
                data=np.array(var_explained[:-1]).reshape(1, -1),
                columns=kname.split("+"),
                index=[o]
            )
            out_info = pd.concat(
                objs=[out_info, new_row]
            )

        # Fill in missing explained boxes with zero
        out_info.fillna(value=0, inplace=True)

        if print_drop_count:
            if feature_name is not None:
                print(f"Number of models dropped because feature not present: {n_feature_drops}")
            print(f"Number of models dropped because of explained threshold not met: {n_explained_drops}")

        # Now plot
        if cluster:
            col_cluster = True
            row_cluster = True
            assert len(out_info.index) > 1, (
                "Not enough models meet criteria (clustermap) requested!"
                f"  (N={len(out_info.index)})"
            )
        else:
            col_cluster = False
            row_cluster = False
            assert len(out_info.index) > 0, (
                "Not enough models meet criteria (heatmap) requested!"
                f" (N={len(out_info.index)})"
            )
        scale_size = 1
        if figsize is None:
            c_unit_h = 0.01
            c_unit_w = 0.01
            c_min_height = .1 * c_unit_h + 1
            c_min_width = .1 * c_unit_w + 1
            c_char_pad = 0.01
            width = max(c_unit_w * out_info.shape[1], c_min_width)
            width += c_char_pad * max(list(map(len, out_info.index.tolist())))
            height = max(c_unit_h * out_info.shape[0], c_min_height)
            height += c_char_pad * max(list(map(len, out_info.columns.tolist())))
            figsize = (width*scale_size, height*scale_size)

        clm = sns.clustermap(
            out_info.transpose(),
            figsize=figsize,
            annot=show_vals,
            annot_kws={'size': 6*scale_size},
            robust=True,
            cmap="Greens",
            # cbar_pos=(0.8, 0.8, 0.05, 0.15),
            fmt="g",
            dendrogram_ratio=(0.05, 0.05),
            col_cluster=col_cluster,
            row_cluster=row_cluster,
            **clustermap_kwargs
            #cbar=False
        )
        #clm.ax_row_dendrogram.set_visible(False)
        #clm.ax_col_dendrogram.set_visible(False)
        #clm.cax.set_visible(False)
        ax = clm.ax_heatmap
        # else:
        #     fig, ax = plt.subplots(1, 1, figsize=figsize)
        #     clm = sns.heatmap(
        #         out_info.transpose(),
        #         ax=ax,
        #         annot=show_vals,
        #         annot_kws={'size': 6},
        #         vmin=0,
        #         vmax=1,
        #         cmap="Greens",
        #         **kwargs
        #     )
        # Adjust text for easier reading
        plt.setp(
            ax.xaxis.get_majorticklabels(),
            rotation=45,
            horizontalalignment="right"
        )
        # Add text if requested
        if show_vals:
            for t in ax.texts:
                if float(t.get_text()) > 0:
                    t.set_text(t.get_text())
                else:
                    t.set_text("")
        # Set xlabel on the heatmap axes
        ax.set_xlabel('Omics features', fontweight='bold', fontsize=8*scale_size)
        ax.set_ylabel('Dynamics ', fontweight='bold', fontsize=8*scale_size)
        ax.get_xaxis().set_tick_params(which='both', labelsize=6*scale_size)
        ax.get_yaxis().set_tick_params(which='both', labelsize=6*scale_size)
        ax.set_title("Explained variation", fontweight='bold', fontsize=9*scale_size, loc='left')
        return clm

    def plot_parts(
        self,
        out_label,
        x_axis_label,
        reverse_transform_axes=False,
        **kwargs
    ):
        """Plot independent kernel components.

        Parameters
        ----------

        Attributes
        ----------

        """

        pkp = self.models[out_label].plot_parts(
            x_idx=self.feat_names.index(x_axis_label),
            # unit_idx=self.unit_idx,
            col_names=self.feat_names,
            lik=self.likelihood,
            categorical_dict=self.categorical_dict,
            data=(self.X.values, self.Y[out_label].values.reshape(-1, 1)),
            **kwargs,
        )

        # Reverse transform if requested
        if reverse_transform_axes is True:
            if hasattr(self, "X_stds"):

                for a in pkp[1].flatten():

                    # Get feature name
                    xlab_name = a.get_xlabel()

                    # # Pass over residual plot for x-axis transform
                    if "fitted" in xlab_name:
                        break

                    ticks_loc = a.get_xticks().tolist()
                    a.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                    a.set_xticklabels(
                        self.reverse_transform(
                            array=ticks_loc,
                            feature_name=xlab_name,
                            input_type="X"
                        )
                    )

            if hasattr(self, "Y_stds"):
                for a in pkp[1].flatten():

                    # Pass over residual plot for x-axis transform
                    if "fitted" in xlab_name:
                        ticks_loc = a.get_xticks().tolist()
                        a.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                        a.set_yticklabels(
                            self.reverse_transform(
                                array=ticks_loc,
                                feature_name=out_label,
                                input_type="Y"
                            )
                        )
                    else:
                        ticks_loc = a.get_yticks().tolist()
                        a.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                        a.set_yticklabels(
                            self.reverse_transform(
                                array=ticks_loc,
                                feature_name=out_label,
                                input_type="Y"
                            )
                        )

        return pkp

    def plot_feature_metrics(
        self,
        feature_name,
        print_drop_count=False,
        return_df=False,
        top_n=None,
        min_total_explained=0.8
    ):

        # Drop counters
        n_feature_drops = 0
        n_explained_drops = 0

        out_names_list = []
        out_values_list = []

        # Loop through all models
        for o in self.out_names:
            # print(f"{o=}")

            # Copy model
            m_copy = gpflow.utilities.deepcopy(self.models[o])
            var_explained = m_copy.variance_explained

            if 1 - var_explained[-1] < min_total_explained:
                continue

            # First see if a feature is in each model
            if feature_name is not None:

                # Get index of specified feature
                feature_index = np.where(self.X.columns == feature_name)[0][0]
                # Figure out where it is in the model kernel
                feature_kernel_flags = [
                    str(feature_index) in y for y in [
                        re.findall(r"\[(\d+)\]", x)
                        for x in m_copy.kernel_name.split("+")
                        ]
                    ]
                # print(f"{feature_index=}, {feature_kernel_flags=}")

                # If this feature is in the selected model and the model has other compon
                if sum(feature_kernel_flags) > 0:
                    out_values_list.append(
                        max(np.array(m_copy.variance_explained[:-1])[np.array(feature_kernel_flags)])
                    )
                    out_names_list.append(o)

                # If this feature doesn't exist in the model then pass
                elif sum(feature_kernel_flags) == 0:
                    n_feature_drops += 1
                    # Don't need to check variance explained anymore
                    continue
            else:
                feature_index = None

        if print_drop_count:
            if feature_name is not None:
                print(f"Number of models dropped because feature not present: {n_feature_drops}")
            print(f"Number of models dropped because of explained threshold not met: {n_explained_drops}")

        # Order output
        metric_df = pd.DataFrame(
            data={
                "name": out_names_list,
                "metric": out_values_list
            }
        ).sort_values("metric", ascending=False)

        # Truncate output
        if top_n is not None:
            metric_df = metric_df.head(top_n)

        if return_df:
            return metric_df
        else:
            # Plot
            p = sns.barplot(
                data=metric_df,
                y="name",
                x="metric"
            )
            return p

    def plot_marginal(
        self,
        out_label,
        x_axis_label,
        unit_label=None,
        num_funs=100,
        ax=None,
        plot_points=True,
        reverse_transform_axes=False,
        **kwargs,
    ):
        # Pull off specific model from trained models
        m = self.models[out_label]

        # Also get index of x axis variable
        x_idx = self.feat_names.index(x_axis_label)
        y_idx = self.out_names.index(out_label)

        # if self.model_selection_type == "penalized":
        gpf = m.plot_functions(
            data=convert_data_to_tensors(
                self.X.to_numpy(),
                self.Y[out_label].to_numpy().reshape(-1, 1)
            ),
            x_idx=x_idx,
            col_names=self.feat_names,
            unit_idx=self.unit_idx,
            unit_label=unit_label,
            num_funs=num_funs,
            ax=ax,
            plot_points=plot_points,
            **kwargs,
        )

        # Reverse transform if requested
        if reverse_transform_axes is True:
            if hasattr(self, "X_stds"):
                
                # for a in gpf[1].flatten():

                # Get feature name
                xlab_name = gpf.get_xlabel()

                # # Pass over residual plot for x-axis transform
                # if "fitted" in xlab_name:
                #     break

                ticks_loc = gpf.get_xticks().tolist()
                gpf.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                gpf.set_xticklabels(
                    self.reverse_transform(
                        array=ticks_loc,
                        feature_name=xlab_name,
                        input_type="X"
                    )
                )

            if hasattr(self, "Y_stds"):
                # for a in gpf[1].flatten():

                # # Pass over residual plot for x-axis transform
                # if "fitted" in xlab_name:
                #     ticks_loc = a.get_xticks().tolist()
                #     a.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                #     a.set_yticklabels(
                #         self.reverse_transform(
                #             array=ticks_loc,
                #             feature_name=out_label,
                #             input_type="Y"
                #         )
                #     )
                # else:
                ticks_loc = gpf.get_yticks().tolist()
                gpf.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                gpf.set_yticklabels(
                    self.reverse_transform(
                        array=ticks_loc,
                        feature_name=out_label,
                        input_type="Y"
                    )
                )

        return gpf

    def reverse_transform(
        self,
        array,
        feature_name=None,
        input_type="X",
        round_digits=1
    ):
        """ Return input values on original scale.
        """

        if input_type == "X":
            assert hasattr(
                self, "X_stds"
            ), "Standardize_X wasn't called in GPSearch()"

            if feature_name is None:
                scale_vals = self.X_stds.values
                shift_vals = self.X_means.values
            else:
                scale_vals = self.X_stds[feature_name]
                shift_vals = self.X_means[feature_name]

        elif input_type == "Y":
            assert hasattr(
                self, "Y_stds"
            ), "Y_transform wasn't called in GPSearch()"

            if feature_name is None:
                scale_vals = self.Y_stds.values
            else:
                scale_vals = self.Y_stds[feature_name]

            if hasattr(self, "Y_means"):
                if feature_name is None:
                    shift_vals = self.Y_means.values
                else:
                    shift_vals = self.Y_means[feature_name]
            else:
                shift_vals = np.zeros_like(scale_vals)
        else:
            raise ValueError("Unknown type requested for transform!")

        # Perform transformation
        array_transformed = np.round(
            scale_vals * np.array(array) + shift_vals,
            decimals=round_digits
        )

        return array_transformed


def kernel_test(
    X,
    Y,
    k,
    mean_function=gpflow.mean_functions.Constant(),
    num_restart=5,
    random_init=True,
    random_seed=None,
    verbose=False,
    likelihood="gaussian",
    scale_value=None,
    use_priors=True,
    keep_data=False,
    X_holdout=None,
    Y_holdout=None,
    split=False,
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
    # Specify model structure
    best_model = PSVGP(
        X=X,
        Y=Y,
        mean_function=mean_function,
        kernel=k,
        verbose=verbose,
        penalized_options={"penalization_factor": 0.0},
        sparse_options={},
        variational_options={
            "likelihood": likelihood,
            # "scale_value": scale_value
            # "variational_priors": use_priors
        }
    )

    if random_init and num_restart > 1:
        best_model.random_restart_optimize(
            data=convert_data_to_tensors(X, Y),
            num_restart=num_restart,
            randomize_kwargs={
                "random_seed": random_seed
            }
        )
    elif num_restart > 1:
        best_model.random_restart_optimize(
            data=convert_data_to_tensors(X, Y),
            num_restart=num_restart,
            randomize_kwargs={
                "scale": 0.0,
                "random_seed": random_seed
            }
        )
    else:
        best_model.optimize_params(
            data=convert_data_to_tensors(X, Y)
        )

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
        estimated_loglik = best_model.log_posterior_density(
            data=(X, Y)
        ).numpy()

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


def set_feature_kernels(f, kern_list, cat_vars):
    if f in cat_vars:
        k_list = [Categorical(active_dims=[f])]
    else:
        k_list = kern_list.copy()
        for k_ in k_list:
            k_.active_dims = np.array([f])
    return k_list


def loc_kernel_search(
    X,
    Y,
    kern_list,
    base_kern=None,
    base_name=None,
    cat_vars=[],
    depth=0,
    operation="sum",
    prod_index=None,
    prev_models=None,
    lik="gaussian",
    scale_value=None,
    verbose=False,
    num_restart=5,
    random_seed=None,
    X_holdout=None,
    Y_holdout=None,
    split=False,
):
    """
    This function performs the local kernel search.
    """

    bic_dict = {}

    if verbose:
        print(f"Base kernel: {base_name}")

    # Search over features in X
    for f in np.arange(X.shape[1]):
        if verbose:
            print(f"Working on feature {f} now")

        temp_kern_list = [gpflow.utilities.deepcopy(x) for x in kern_list]
        # Set kernel list based on feature currently searching
        k_list = set_feature_kernels(
            f=f, kern_list=temp_kern_list, cat_vars=cat_vars
        )
        # Add no /static/ kernel to test if first level and first feature
        if f == 0 and depth == 1:  # and lik=='gaussian':
            # print(f'Current list of kernels: {k_list}')
            empty_kernel = gpflow.kernels.Constant(variance=f64(1e-6))
            set_trainable(empty_kernel.variance, False)
            k_list += [empty_kernel]
            # k_list += [Empty()]

        # Search over kernels
        for k in k_list:
            # Get kernel name and dimension
            k_info = (
                k.name + str(k.active_dims) if k.name != "constant" else k.name
            )
            if verbose:
                print("Current kernel being tested: {}".format(k_info))

            # Update kernel information with base if possible
            if base_kern is not None:
                # Make copy of base_kern
                base_kern_ = gpflow.utilities.deepcopy(base_kern)

                # Set parameters back to zero
                for p in base_kern_.trainable_parameters:
                    p.assign(1.0)

                if operation == "sum":
                    # Skip operation if categorical kernel exists
                    if "categorical[" + str(f) + "]" in base_name:
                        continue
                    #    # print('Sum kernel being performed.')
                    try:
                        # Get order correct
                        if base_name < k_info:
                            k = gpflow.kernels.Sum(kernels=[base_kern_, k])
                            k_info = base_name + "+" + k_info
                        else:
                            k = gpflow.kernels.Sum(kernels=[k, base_kern_])
                            k_info = k_info + "+" + base_name

                        # Make sure this is something that hasn't been tested
                        # yet
                        if check_if_model_exists(k_info, prev_models):
                            continue

                        m, bic = kernel_test(
                            X,
                            Y,
                            k,
                            likelihood=lik,
                            scale_value=scale_value,
                            verbose=verbose,
                            num_restart=num_restart,
                            random_seed=random_seed,
                            X_holdout=X_holdout,
                            Y_holdout=Y_holdout,
                            split=split,
                        )

                        bic_dict[k_info] = {
                            "kernel": k,
                            "model": m,
                            "bic": bic,
                            "depth": depth,
                            "parent": base_name,
                            "try_next": True,
                        }
                    except Exception:
                        None

                elif operation == "product":
                    # Skip operation if categorical kernel exists
                    if "categorical[" + str(f) + "]" in base_name:
                        continue
                    #     #print('Product kernel being performed.')

                    # Skip if already a product
                    # (only allow two-way interactions)
                    if "*" in base_name:
                        continue

                    try:
                        # Set new variance to 1, don't train
                        try:
                            set_trainable(k.variance, False)
                        except Exception:
                            set_trainable(k.base_kernel.variance, False)

                        # Get order correct
                        if base_name < k_info:
                            k = gpflow.kernels.Product(kernels=[base_kern_, k])
                            k_info = base_name + "*" + k_info
                        else:
                            k = gpflow.kernels.Product(kernels=[k, base_kern_])
                            k_info = k_info + "*" + base_name

                        # Make sure this is something that hasn't been tested
                        # yet
                        if check_if_model_exists(k_info, prev_models):
                            continue

                        m, bic = kernel_test(
                            X,
                            Y,
                            k,
                            likelihood=lik,
                            scale_value=scale_value,
                            verbose=verbose,
                            num_restart=num_restart,
                            random_seed=random_seed,
                            X_holdout=X_holdout,
                            Y_holdout=Y_holdout,
                            split=split,
                        )
                        #                             print(k_info)
                        # bic_dict[k_info] = [k, m, bic, depth, base_name]
                        bic_dict[k_info] = {
                            "kernel": k,
                            "model": m,
                            "bic": bic,
                            "depth": depth,
                            "parent": base_name,
                            "try_next": True,
                        }
                    except Exception:
                        None

                elif operation == "split_product":
                    # print('Split product being tested.')
                    # Set new variance to 1, don't train
                    # (need base if periodic)
                    try:
                        set_trainable(k.variance, False)
                    except Exception:
                        set_trainable(k.base_kernel.variance, False)
                    new_dict = prod_kernel_creation(
                        X=X,
                        Y=Y,
                        base_kernel=base_kern_,
                        base_name=base_name,
                        new_kernel=k,
                        prev_models=prev_models,
                        depth=depth,
                        lik=lik,
                        scale_value=scale_value,
                        num_restart=num_restart,
                        random_seed=random_seed,
                        X_holdout=X_holdout,
                        Y_holdout=Y_holdout,
                        split=split,
                    )
                    bic_dict.update(new_dict)
            else:
                m, bic = kernel_test(
                    X,
                    Y,
                    k,
                    likelihood=lik,
                    scale_value=scale_value,
                    verbose=verbose,
                    num_restart=num_restart,
                    random_seed=random_seed,
                    X_holdout=X_holdout,
                    Y_holdout=Y_holdout,
                    split=split,
                )
                # bic_dict[k_info] = [k, m, bic, depth, 'None']
                bic_dict[k_info] = {
                    "kernel": k,
                    "model": m,
                    "bic": bic,
                    "depth": depth,
                    "parent": "None",
                    "try_next": True,
                }

    return bic_dict


def prod_kernel_creation(
    X,
    Y,
    base_kernel,
    base_name,
    new_kernel,
    depth,
    lik,
    scale_value=None,
    verbose=False,
    num_restart=5,
    random_seed=None,
    prev_models=[],
    X_holdout=None,
    Y_holdout=None,
    split=False,
):
    """
    Produce kernel for product operation
    """

    bic_dict = {}

    for feat in range(len(base_kernel.kernels)):
        temp_kernel = gpflow.utilities.deepcopy(base_kernel)
        # temp_kernel.kernels[feat] = temp_kernel.kernels[feat] * new_kernel
        temp_name = base_name.split("+")
        k_info = new_kernel.name + str(new_kernel.active_dims)
        # Skip operation if categorical kernel exists
        # print('k_info:',k_info,'temp_name:',temp_name[feat])
        if "categorical" + str(new_kernel.active_dims) not in temp_name[feat]:
            # print('Testing feature')

            # Only allow two-way interactions currently
            if "*" in temp_name[feat]:
                continue

            try:
                # Get order correct of product term, and higher order correct
                # with the overall sum term
                if temp_name[feat] < k_info:
                    temp_name[feat] = temp_name[feat] + "*" + k_info
                    temp_kernel.kernels[feat] = gpflow.kernels.Product(
                        kernels=[temp_kernel.kernels[feat], new_kernel]
                    )

                else:
                    temp_kernel.kernels[feat] = gpflow.kernels.Product(
                        kernels=[new_kernel, temp_kernel.kernels[feat]]
                    )

                    # Insert new starting product kernel in the right sum place
                    # print('Figuring out indexing')
                    # print(f'temp_name: {temp_name}')
                    # print(np.where([k_info < x for x in temp_name]))
                    try:
                        new_idx = np.where([k_info < x for x in temp_name])[0][
                            0
                        ]
                    except Exception:
                        new_idx = len(temp_name) - 1
                    cur_component_name = temp_name.pop(feat)
                    # print(f'new_idx: {new_idx}')
                    # print(f'cur_component_name: {cur_component_name}')
                    cur_component_name = k_info + "*" + cur_component_name
                    temp_name.insert(new_idx, cur_component_name)
                    # print(f'temp_name: {temp_name}')

                # Join everything back together
                k_info = "+".join(temp_name)
                # print(f'k_info: {k_info}')

                # Make sure this is something that hasn't been tested yet
                if check_if_model_exists(k_info, prev_models):
                    continue
                # print('fitting model')
                m, bic = kernel_test(
                    X,
                    Y,
                    temp_kernel,
                    likelihood=lik,
                    scale_value=scale_value,
                    verbose=verbose,
                    num_restart=num_restart,
                    random_seed=random_seed,
                    X_holdout=X_holdout,
                    Y_holdout=Y_holdout,
                    split=split,
                )
                # print(k_info)
                # bic_dict[k_info] = [temp_kernel, m, bic, depth, base_name]
                bic_dict[k_info] = {
                    "kernel": temp_kernel,
                    "model": m,
                    "bic": bic,
                    "depth": depth,
                    "parent": base_name,
                    "try_next": True,
                }

            except Exception as e:
                print(e)
                print(f"Error with product kernel test {k_info}")
                None

    return bic_dict


def check_if_better_metric(model_dict, depth):  # previous_dict, current_dict):
    """
    Check to see if a better metric was found in current layer,
    otherwise end search.
    """

    prev_vals = [
        x["bic"] for x in model_dict.values() if x["depth"] == (depth - 1)
    ]
    new_vals = [x["bic"] for x in model_dict.values() if x["depth"] == (depth)]
    if len(prev_vals) > 0 and len(new_vals) > 0:
        best_prev = min(prev_vals)
        best_new = min(new_vals)
    else:
        return False

    return True if best_new < best_prev else False


def keep_top_k(res_dict, depth, metric_diff=6, split=False):
    best_bic = np.Inf
    out_dict = res_dict.copy()

    # Specify transform function if needed
    if split:

        def t_func(x):
            return np.log(x)  # np.exp(x)

    else:

        def t_func(x):
            return x

    # Find results from the current depth
    best_bic = min(
        [v["bic"] for k, v in res_dict.items() if v["depth"] == depth]
    )
    #     for k,v in res_dict.items():
    #         if v[3] == depth and v[2] < best_bic:

    # Only keep results in diff window
    for k, v in res_dict.items():
        if v["depth"] == depth and v["bic"] - best_bic > t_func(metric_diff):
            v["try_next"] = False
            # out_dict.pop(k)

    return out_dict


def prune_best_model(
    res_dict,
    depth,
    lik,
    scale_value=None,
    verbose=False,
    num_restart=5,
    random_seed=None
):
    out_dict = res_dict.copy()

    # Get best model from current depth
    best_bic, best_model_name, best_model = min(
        [(i["bic"], k, i["model"]) for k, i in res_dict.items()]
    )
    #     print(f'Best model: {best_model_name}')

    # Split kernel name to sum pieces
    kernel_names = best_model_name.split("+")

    # Loop through each kernel piece and prune out refit and evaluate
    if len(kernel_names) <= 1:
        #         print('No compound terms to prune!')
        return res_dict

    for i in range(len(kernel_names)):
        # k = gpflow.kernels.Sum(
        #     kernels = [k_ for i_, k_ in enumerate(best_model.kernel.kernels)
        #                if i_ != i])
        k_info = "+".join(
            [x_ for i_, x_ in enumerate(kernel_names) if i_ != i]
        )
        kerns = [
            k_ for i_, k_ in enumerate(best_model.kernel.kernels) if i_ != i
        ]
        if len(kerns) > 1:
            k = gpflow.kernels.Sum(kernels=kerns)
        else:
            k = kerns[0]

        # Check to see if this model has already been fit previously
        if check_if_model_exists(k_info, list(res_dict.keys())):
            continue

        m, bic = kernel_test(
            best_model.data[0],
            best_model.data[1],
            k,
            likelihood=lik,
            scale_value=scale_value,
            verbose=verbose,
            num_restart=num_restart,
            random_seed=random_seed
        )
        # If better model found then save it
        if bic < best_bic:
            #             print(f'New better model found: {k_info}')
            # out_dict[k_info] = [k, m, bic, depth, best_model_name]
            out_dict[k_info] = {
                "kernel": m.kernel,
                "model": m,
                "bic": bic,
                "depth": depth,
                "parent": best_model_name,
                "try_next": True,
            }
    return out_dict


def prune_best_model2(
        res_dict,
        depth,
        lik,
        scale_value=None,
        verbose=False,
        num_restart=5,
        random_seed=None
    ):
    out_dict = res_dict.copy()

    # Get best model from current depth
    best_bic, best_model_name, best_model = min(
        [
            (i["bic"], k, i["model"])
            for k, i in res_dict.items()
            if i["depth"] == depth
        ]
    )
    # print(f'Best model: {best_model_name}')

    # Copy so as not to overwrite best current model
    best_model = gpflow.utilities.deepcopy(best_model)

    # Split kernel name to sum pieces
    kernel_names = re.split("\+", best_model_name)
    # print(kernel_names)

    # Loop through each kernel piece and prune out refit and evaluate
    if len(kernel_names) <= 1 and "*" not in kernel_names[0]:
        # print('No compound terms to prune!')
        return res_dict

    for i in range(len(kernel_names)):
        if verbose:
            print(f"Current kernel component: {kernel_names[i]}")
        # Glue together the kernel pieces that are not currently being pruned
        k_info = "+".join(
            [x_ for i_, x_ in enumerate(kernel_names) if i_ != i]
        )
        kerns = [
            k_ for i_, k_ in enumerate(best_model.kernel.kernels) if i_ != i
        ]

        # TODO: Still can't figure out product term issue
        # Check if this term is a product term, add to end if so
        if "*" in kernel_names[i]:
            if verbose:
                print("Currently dealing with product terms!")

            # Deal with a single product term versus multiple terms
            if len(kernel_names) == 1:
                prod_kernel = best_model.kernel
                other_kernel = None
                other_name = ""
            else:
                prod_kernel = best_model.kernel.kernels[i]
                other_kernel = kerns
                other_name = k_info

            out_dict = prune_prod_kernel(
                prod_kernel=prod_kernel,
                prod_name=kernel_names[i],
                res_dict=out_dict,
                best_model=best_model,
                best_bic=best_bic,
                best_model_name=best_model_name,
                depth=depth,
                other_kernel=other_kernel,
                other_name=other_name,
                lik=lik,
                scale_value=scale_value,
                verbose=verbose,
                num_restart=num_restart,
                random_seed=random_seed
            )

            # Skip the rest of the iteration if product was involved
            continue

        if len(kerns) > 1:
            k = gpflow.kernels.Sum(kernels=kerns)
        else:
            k = kerns[0]

        # Check to see if this model has already been fit previously
        if check_if_model_exists(k_info, list(res_dict.keys())):
            continue
        else:
            # Reset parameters
            for p in k.trainable_parameters:
                p.assign(f64(1))
            m, bic = kernel_test(
                best_model.data[0],
                best_model.data[1],
                k,
                likelihood=lik,
                scale_value=scale_value,
                verbose=verbose,
                num_restart=num_restart,
                random_seed=random_seed
            )
        # If better model found then save it
        if bic < best_bic:
            if verbose:
                print(f"New better model found: {k_info}")
            # out_dict[k_info] = [k, m, bic, depth, best_model_name]
            out_dict[k_info] = {
                "kernel": m.kernel,
                "model": m,
                "bic": bic,
                "depth": depth,
                "parent": best_model_name,
                "try_next": True,
            }
    return out_dict


def prune_prod_kernel(
    prod_kernel,
    prod_name,
    res_dict,
    best_model,
    best_bic,
    best_model_name,
    depth,
    other_kernel=None,
    other_name="",
    lik="gaussian",
    scale_value=None,
    verbose=False,
    num_restart=5,
    random_seed=None,
    **kwargs,
):
    out_dict = res_dict.copy()
    other_kernel = gpflow.utilities.deepcopy(other_kernel)
    prod_kernel = gpflow.utilities.deepcopy(prod_kernel)

    # Split product kernel name
    kernel_parts = prod_name.split("*")

    # Try to check if there are other kernels in the prod piece
    if prod_kernel.name != "product":
        if verbose:
            print(f"Prod kernel issues with {prod_kernel}. Exiting.\n")
        return out_dict

    # Loop through each kernel piece
    for i in range(len(prod_kernel.kernels)):
        # Get new kernel name
        try:
            new_piece = kernel_parts[i]
        except IndexError:
            print(f"IndexError with index {i} in kernel_parts {kernel_parts}")
            return out_dict
        if verbose:
            print(f"New kernel piece being tested: {new_piece}")

        if other_name == "":
            k_info = new_piece
            k = prod_kernel.kernels[i]
        else:
            order_set = np.argsort([other_name, new_piece])
            k_info = "+".join(np.array([other_name, new_piece])[order_set])

            # Get new kernel piece
            # print(other_kernel)
            if not isinstance(other_kernel, list):
                other_kernel = [other_kernel]
            kerns = list(
                np.array(other_kernel + [prod_kernel.kernels[i]])[order_set]
            )
            k = gpflow.kernels.Sum(kernels=kerns)

        if verbose:
            print(f"Model about to be fit: {k_info}")
        # Check to see if kernel has already been tested
        if check_if_model_exists(k_info, list(res_dict.keys())):
            if verbose:
                print(f"{k_info} has already been fit. Skipping!")
            continue

        else:
            # Reset kernel parameters
            for p in k.trainable_parameters:
                p.assign(f64(1))

            # Test kernel if appropriate
            m, bic = kernel_test(
                X=best_model.data[0],
                Y=best_model.data[1],
                k=k,
                likelihood=lik,
                scale_value=scale_value,
                verbose=verbose,
                num_restart=num_restart,
                random_seed=random_seed
            )

            if verbose:
                print(f"model = {k_info}, BIC = {bic}")

        # Save if better kernel
        if bic < best_bic:
            if verbose:
                print(f"Found better kernel! {k_info}")
            out_dict[k_info] = {
                "kernel": m.kernel,
                "model": m,
                "bic": bic,
                "depth": depth,
                "parent": best_model_name,
                "try_next": True,
            }

    return out_dict


def full_kernel_search(
    X,
    Y,
    kern_list,
    cat_vars=[],
    max_depth=5,
    keep_all=False,
    metric_diff=6,
    early_stopping=True,
    prune=True,
    num_restart=5,
    lik="gaussian",
    scale_value=None,
    verbose=False,
    debug=False,
    keep_only_best=True,
    softmax_select=False,
    random_seed=None,
    feature_name=None
):
    """
    This function runs the entire kernel search, calling helpers along the way.
    """

    # Set seed if requested
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create initial dictionaries to insert
    search_dict = {}
    edge_list = []

    # Make sure the inputs are in the correct format
    x_dim = 1
    if len(X.shape) == 2:
        x_dim = X.shape[1]
    X = X.to_numpy().reshape(-1, x_dim)

    if feature_name is None:
        Y = Y.to_numpy().reshape(-1, 1)
        scale_value = (
            None if scale_value is None
            else scale_value.to_numpy().reshape(1, 1)
        )
    else:
        Y = Y[feature_name].to_numpy().reshape(-1, 1)
        scale_value = (
            None if scale_value is None
            else scale_value[feature_name]
        )

    # Flag for missing values
    x_idx = ~np.isnan(X).any(axis=1)
    y_idx = ~np.isnan(Y).flatten()

    # Get complete cases
    comp_X = X[x_idx & y_idx]
    comp_y = Y[x_idx & y_idx]

    X = comp_X
    Y = comp_y

    # Go through each level of depth allowed
    for d in range(1, max_depth + 1):
        if verbose:
            print(f"Working on depth {d} now")
        # If first level then there is no current dictionary to update
        if d == 1:
            search_dict = loc_kernel_search(
                X=X,
                Y=Y,
                kern_list=kern_list,
                cat_vars=cat_vars,
                depth=d,
                lik=lik,
                scale_value=scale_value,
                verbose=debug,
                num_restart=num_restart,
            )
        else:
            # Create temporary dictionary to hold new results
            temp_dict = search_dict.copy()
            for k in search_dict.keys():
                # Make sure to only search through the previous depth
                # and that it is a kernel we want to continue searching down
                # and it isn't constant
                if (
                    search_dict[k]["depth"] != d - 1
                    or search_dict[k]["try_next"] is False
                    or k == "constant"
                ):
                    continue

                # # Make sure to not continue the search for constant kernel
                # if k == 'constant':
                #     continue

                cur_kern = search_dict[k]["kernel"]

                new_res = loc_kernel_search(
                    X=X,
                    Y=Y,
                    kern_list=kern_list,
                    base_kern=cur_kern,
                    base_name=k,
                    cat_vars=cat_vars,
                    depth=d,
                    lik=lik,
                    scale_value=scale_value,
                    operation="sum",
                    prev_models=temp_dict.keys(),
                    verbose=debug,
                    num_restart=num_restart,
                )
                temp_dict.update(new_res)

                # Update graph dictionary
                for k_ in new_res.keys():
                    edge_list += [(k, k_)]

                # For product break up base kernel and push in each possibility
                # else just simple product with single term
                op = "split_product" if cur_kern.name == "sum" else "product"
                new_res = loc_kernel_search(
                    X=X,
                    Y=Y,
                    kern_list=kern_list,
                    base_kern=cur_kern,
                    base_name=k,
                    cat_vars=cat_vars,
                    depth=d,
                    lik=lik,
                    scale_value=scale_value,
                    operation=op,
                    prev_models=temp_dict.keys(),
                    verbose=debug,
                    num_restart=num_restart,
                )
                temp_dict.update(new_res)

                # Update graph dictionary
                for k_ in new_res.keys():
                    edge_list += [(k, k_)]

            # found_better = check_if_better_metric(search_dict, temp_dict)

            # Overwrite search dictionary with temporary results
            search_dict = temp_dict

        # What is the best model we have right now?
        # [x['bic'] for x in model_dict.values() if x['depth'] == (depth)]

        best_model_name = min(
            [
                (i["bic"], i["depth"], k)
                for k, i in search_dict.items()
                if i["depth"] == d
            ]
        )[2]
        if verbose:
            print(f"Best model for depth {d} is {best_model_name}")

        # Add data back in for the best model
        search_dict[best_model_name]["model"].data = (
            tf.convert_to_tensor(X),
            tf.convert_to_tensor(Y),
        )

        # If the best model is constant then end search
        if best_model_name == "constant":
            if verbose:
                print("Best model is constant, going to stop searching now")
            break

        # Early stopping?
        if early_stopping and d > 1:
            found_better = check_if_better_metric(
                model_dict=search_dict, depth=d
            )

            # If no better kernel found then exit search
            if not found_better:
                if verbose:
                    print("No better kernel found in layer, exiting search!")
                # Prune best model
                if prune:
                    if verbose:
                        print("Pruning now")
                    search_dict = prune_best_model2(
                        search_dict,
                        depth=d,
                        lik=lik,
                        scale_value=scale_value,
                        verbose=verbose,
                        num_restart=num_restart,
                    )

                break
            else:
                None

        # Do we want to filter out results to conitnue searching in the future?
        # Only do this if it isn't the last layer
        if d != max_depth:
            # Filter out results from current depth to just keep best kernel
            # options
            if not keep_all:
                search_dict = keep_top_k(
                    search_dict, depth=d, metric_diff=metric_diff
                )

            # If softmax is model selection option then choose best model
            if softmax_select:
                model_info_list = [
                    (i["bic"], k) for k, i in search_dict.items()
                ]
                model_name_selected = softmax_kernel_selection(
                    bic_list=[x[0] for x in model_info_list],
                    name_list=[x[1] for x in model_info_list],
                )

                # Subset this depth to only model selected
                search_dict_copy = search_dict.copy()
                for k, v in search_dict_copy.items():
                    if v["depth"] == d and k != model_name_selected:
                        v["try_next"] = False
                        # search_dict.pop(k)

        # Add data back in for the best model
        best_model_name = min(
            [
                (i["bic"], i["depth"], k)
                for k, i in search_dict.items()
                if i["depth"] == d
            ]
        )[2]
        search_dict[best_model_name]["model"].data = (
            tf.convert_to_tensor(X),
            tf.convert_to_tensor(Y),
        )

        # Prune best model
        if prune:
            if verbose:
                print("Pruning now")
            search_dict = prune_best_model2(
                search_dict,
                depth=d,
                lik=lik,
                scale_value=scale_value,
                verbose=verbose,
                num_restart=num_restart,
            )

        if verbose:
            # # Look for best model
            # best_model_name = min([(i['bic'], i['depth'], k)
            #                        for k, i in search_dict.items()])[2]
            # print(f'Best model for depth {d} is {best_model_name}')
            if d == max_depth:
                print("Reached max depth, ending search.")
            else:
                print("-----------\n")

    # Look for best model
    # best_model_name = min([(i[2], k) for k, i in search_dict.items()])[1]
    best_model_name = min(
        [(i["bic"], i["depth"], k) for k, i in search_dict.items()]
    )[2]
    if verbose:
        print(f"Best model for depth {d} is {best_model_name}")

    # Add data back in for the best model
    search_dict[best_model_name]["model"].data = (
        tf.convert_to_tensor(X),
        tf.convert_to_tensor(Y),
    )

    # Calc R2 of best model
    var_percent = calc_rsquare(
        search_dict[best_model_name]["model"]  # ,
        # best_model_name,
        # lik=lik
    )

    # Keep only the final best model?
    if keep_only_best:
        search_dict = {best_model_name: search_dict[best_model_name]}

    # Return output
    return {
        "models": search_dict,
        "edges": edge_list,
        "best_model": best_model_name,
        "var_exp": var_percent,
    }


def split_kernel_search(
    X,
    Y,
    kern_list,
    unit_idx,
    training_percent=0.7,
    cat_vars=[],
    max_depth=5,
    keep_all=False,
    metric_diff=1,
    early_stopping=True,
    prune=True,
    num_restart=5,
    lik="gaussian",
    scale_value=None,
    verbose=False,
    debug=False,
    keep_only_best=True,
    softmax_select=False,
    random_seed=None,
):
    """
    This function runs the entire kernel search, calling helpers along the way.
    It is different from full_kernel_search() because it splits the data based
    on unit id into training and holdout for evaluation purposes.
    """

    # Set seed if requested
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create initial dictionaries to insert
    search_dict = {}
    edge_list = []

    # Make sure the inputs are in the correct format
    X = X.to_numpy().reshape(-1, X.shape[1])
    Y = Y.to_numpy().reshape(-1, 1)

    # Flag for missing values
    x_idx = ~np.isnan(X).any(axis=1)
    y_idx = ~np.isnan(Y).flatten()

    # Get complete cases
    comp_X = X[x_idx & y_idx]
    comp_y = Y[x_idx & y_idx]

    X = comp_X
    Y = comp_y

    # Subset data
    unique_ids = np.unique(X[:, unit_idx])
    train_ids = np.random.choice(
        unique_ids,
        size=round(training_percent * len(unique_ids)),
        replace=False,
    )
    Y_holdout = Y[np.isin(X[:, unit_idx], train_ids, invert=True)]
    Y = Y[np.isin(X[:, unit_idx], train_ids)]
    X_holdout = X[np.isin(X[:, unit_idx], train_ids, invert=True), :]
    X = X[np.isin(X[:, unit_idx], train_ids), :]

    # Go through each level of depth allowed
    for d in range(1, max_depth + 1):
        if verbose:
            print(f"Working on depth {d} now")
        # If first level then there is no current dictionary to update
        if d == 1:
            search_dict = loc_kernel_search(
                X=X,
                Y=Y,
                kern_list=kern_list,
                cat_vars=cat_vars,
                depth=d,
                lik=lik,
                scale_value=scale_value,
                verbose=debug,
                num_restart=num_restart,
                X_holdout=X_holdout,
                Y_holdout=Y_holdout,
                split=True,
            )
        else:
            # Create temporary dictionary to hold new results
            temp_dict = search_dict.copy()
            for k in search_dict.keys():
                # Make sure to only search through the previous depth
                # and that it is a kernel we want to continue searching down
                # and it isn't constant
                if (
                    search_dict[k]["depth"] != d - 1
                    or search_dict[k]["try_next"] is False
                    or k == "constant"
                ):
                    continue

                # # Make sure to not continue the search for constant kernel
                # if k == 'constant':
                #     continue

                cur_kern = search_dict[k]["kernel"]

                new_res = loc_kernel_search(
                    X=X,
                    Y=Y,
                    kern_list=kern_list,
                    base_kern=cur_kern,
                    base_name=k,
                    cat_vars=cat_vars,
                    depth=d,
                    lik=lik,
                    scale_value=scale_value,
                    operation="sum",
                    prev_models=temp_dict.keys(),
                    verbose=debug,
                    num_restart=num_restart,
                    X_holdout=X_holdout,
                    Y_holdout=Y_holdout,
                    split=True,
                )
                temp_dict.update(new_res)

                # Update graph dictionary
                for k_ in new_res.keys():
                    edge_list += [(k, k_)]

                # For product break up base kernel and push in each possibility
                # else just simple product with single term
                op = "split_product" if cur_kern.name == "sum" else "product"
                new_res = loc_kernel_search(
                    X=X,
                    Y=Y,
                    kern_list=kern_list,
                    base_kern=cur_kern,
                    base_name=k,
                    cat_vars=cat_vars,
                    depth=d,
                    lik=lik,
                    scale_value=scale_value,
                    operation=op,
                    prev_models=temp_dict.keys(),
                    verbose=debug,
                    num_restart=num_restart,
                    X_holdout=X_holdout,
                    Y_holdout=Y_holdout,
                    split=True,
                )
                temp_dict.update(new_res)

                # Update graph dictionary
                for k_ in new_res.keys():
                    edge_list += [(k, k_)]

            # found_better = check_if_better_metric(search_dict, temp_dict)

            # Overwrite search dictionary with temporary results
            search_dict = temp_dict

        # Early stopping?
        if early_stopping and d > 1:
            found_better = check_if_better_metric(
                model_dict=search_dict, depth=d
            )

            # If no better kernel found then exit search
            if not found_better:
                if verbose:
                    print("No better kernel found in layer, exiting search!")
                # Prune best model
                if prune:
                    if verbose:
                        print("Pruning now")
                    search_dict = prune_best_model2(
                        search_dict,
                        depth=d,
                        lik=lik,
                        scale_value=scale_value,
                        verbose=verbose,
                        num_restart=num_restart,
                    )

                break
            else:
                None

        # Do we want to filter out results to conitnue searching in the future?
        # Only do this if it isn't the last layer
        if d != max_depth:
            # Filter out results from current depth to just keep best kernel
            # options
            if not keep_all:
                search_dict = keep_top_k(
                    search_dict, depth=d, metric_diff=metric_diff, split=True
                )

            # If softmax is model selection option then choose best model
            if softmax_select:
                model_info_list = [
                    (i["bic"], k) for k, i in search_dict.items()
                ]
                model_name_selected = softmax_kernel_selection(
                    bic_list=[x[0] for x in model_info_list],
                    name_list=[x[1] for x in model_info_list],
                )

                # Subset this depth to only model selected
                search_dict_copy = search_dict.copy()
                for k, v in search_dict_copy.items():
                    if v["depth"] == d and k != model_name_selected:
                        v["try_next"] = False
                        # search_dict.pop(k)

        else:  # d == max_depth
            # Prune best model
            if prune:
                if verbose:
                    print("Pruning now")
                search_dict = prune_best_model2(
                    search_dict,
                    depth=d,
                    lik=lik,
                    scale_value=scale_value,
                    verbose=verbose,
                    num_restart=num_restart,
                )

        if verbose:
            if d == max_depth:
                print("Reached max depth, ending search.")
            else:
                print("-----------\n")

    # Look for best model
    # best_model_name = min([(i[2], k) for k, i in search_dict.items()])[1]
    best_model_name = min(
        [(i["bic"], i["depth"], k) for k, i in search_dict.items()]
    )[2]
    if verbose:
        print(best_model_name)

    # Variance decomposition of best model
    var_percent = calc_rsquare(
        search_dict[best_model_name]["model"]  # ,
        # best_model_name,
        # lik=lik
    )

    # Keep only the final best model?
    if keep_only_best:
        search_dict = {best_model_name: search_dict[best_model_name]}

    # Return output
    return {
        "models": search_dict,
        "edges": edge_list,
        "best_model": best_model_name,
        "var_exp": var_percent,
        "X_holdout": X_holdout,
        "Y_holdout": Y_holdout,
        "X": X,
        "Y": Y,
    }


def softmax_kernel_selection(bic_list, name_list):
    """
    Takes in BICs, normalizes, and returns an option
    """

    # print(bic_list)
    # print(name_list)

    # Set seed for reproducibility
    # np.random.seed(9102)

    # Filter out "bad" models
    # print(bic_list)
    name_list = [
        name_list[x] for x in range(len(bic_list)) if bic_list[x] != np.Inf
    ]
    bic_list = [x for x in bic_list if x != np.Inf]

    # If there is just a single model then return that one
    if len(bic_list) == 1:
        return name_list[0]

    # Negate values because lower BIC is better
    bic_list = np.array([-x for x in bic_list])

    # Standardize the values to between 0 and 1
    norm_bic_list = (bic_list - min(bic_list)) / (
        max(bic_list) - min(bic_list)
    )

    # Make a probability distribution
    prob_list = np.exp(norm_bic_list) / sum(np.exp(norm_bic_list))

    # Select one model
    model_selected = np.random.choice(a=np.arange(len(prob_list)), p=prob_list)

    return name_list[model_selected]


def softmax_kernel_search(
    X,
    Y,
    kern_list,
    num_trials=5,
    cat_vars=[],
    max_depth=5,
    lik="gaussian",
    verbose=False,
):
    # Set seed
    # np.random.seed(9102)

    # Set return variables
    best_bic = np.Inf
    best_search_dict = {}
    best_edge_list = []
    best_final_name = ""
    best_var_percent = []
    search_book = {}

    # Run through kernel space for a number of trials using softmax exploration
    for i in range(num_trials):
        search_dict, edge_list, best_name, var_percent = full_kernel_search(
            X=X,
            Y=Y,
            kern_list=kern_list,
            cat_vars=cat_vars,
            max_depth=max_depth,
            keep_all=True,
            early_stopping=False,
            prune=False,
            lik=lik,
            verbose=verbose,
            keep_only_best=False,  # True,
            softmax_select=True,
        )

        search_book[i] = search_dict

        print(best_name)
        print(search_dict[best_name][2])
        # Check if BIC is better
        if search_dict[best_name][2] < best_bic:
            print(f"Better model! {best_name}: {search_dict[best_name][2]}")
            best_bic = search_dict[best_name][2]
            best_search_dict = search_dict
            best_edge_list = edge_list
            best_final_name = best_name
            best_var_percent = var_percent

    return (
        best_search_dict,
        best_edge_list,
        best_final_name,
        best_var_percent,
        search_book,
    )
