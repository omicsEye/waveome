import gpflow
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
# Not currently using below because it presents warning on M1 mac
# from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions as tfd
from joblib import Parallel, delayed
from tqdm import tqdm
import copy
from .utilities import (
    gp_likelihood_crosswalk,
    tqdm_joblib,
    find_variance_components,
    calc_bic,
    variance_contributions,
    print_kernel_names
)
from .predictions import (
    pred_kernel_parts,
    gp_predict_fun
)
from .regularization import (
    make_folds
)

class BaseGP(gpflow.models.SVGP):
    ''' Basic Gaussian process that inherits SVGP gpflow structure.

    Attributes
    ----------
    X: numpy.array
    Y: numpy.array
    kernel: gpflow.kernel.Kernel
    verbose: boolean

    Methods
    -------
    
    '''
    def __init__(
            self, 
            X, Y, 
            mean_function=gpflow.utilities.deepcopy(gpflow.mean_functions.Constant()),
            kernel=gpflow.utilities.deepcopy(gpflow.kernels.SquaredExponential()), 
            verbose=False,
            **svgp_kwargs):
        # Set values
        self.X = X
        self.Y = Y
        self.data = (tf.convert_to_tensor(X), tf.convert_to_tensor(Y))
        self.mean_function = mean_function
        self.kernel = kernel
        self.kernel_name = ''
        self.verbose = verbose

        # TODO: Store X and Y as tensor variables maybe to help retracing?

        # Check for missing data
        assert np.isnan(X).sum() == 0, (
            'Missing values in X found. This is currently not allowed!'
        )
        assert np.isnan(Y).sum() == 0, (
            'Missing values in Y found. This is currently not allowed!'
        )

        # Fill in information for parent class
        super().__init__(
            mean_function=mean_function,
            kernel=kernel,
            likelihood=gpflow.likelihoods.Gaussian(),
            inducing_variable=gpflow.inducing_variables.InducingPoints(X),
            **svgp_kwargs
        )

        # Get the string for the kernel name as well
        self.update_kernel_name()

        # Freeze inducing variables for this model type
        gpflow.utilities.set_trainable(
            self.inducing_variable, 
            False
        )


    def update_kernel_name(self):
        try:
            self.kernel_name = print_kernel_names(self.kernel, with_idx=True)
        except TypeError:
            self.kernel_name = print_kernel_names(self.kernel)
        if type(self.kernel_name) not in [list, str]:
            self.kernel_name = '+'.join(list(self.kernel_name))

        return None
    
    def randomize_params(
            self,
            loc=0.,
            scale=1.,
            random_seed=None
    ):
        ''' Randomize model parameters from sampled normal distribution.
        
        Parameters
        ----------

        Returns
        -------
        None
        '''

        # Set random seed
        np.random.seed(random_seed)

        # Go through each trainable variable
        for p in self.trainable_parameters:
            # Sample unconstrained values from a distribution
            # Need to do special stuff for q_std (triangular fill)
            if p.name == 'fill_triangular':
                # Sample from wishart distribution
                # sample_matrix = scipy.stats.wishart(
                #     df=self.X.shape[0],
                #     scale=(
                #         scale
                #         *(1./self.X.shape[0])
                #         *np.eye(self.X.shape[0])
                #     )
                # ).rvs(size=1, random_state=random_seed)

                sample_matrix = np.diag(
                    np.random.exponential(scale=scale, size=(self.X.shape[0]))
                )
                try:
                    p.assign(tf.convert_to_tensor(sample_matrix[None, :, :]))
                except ValueError:
                    if self.verbose: print("Error assigning random values to fill_triangular, skipping!")

            else:            
                unconstrain_vals = np.random.normal(
                    loc=loc,
                    scale=scale,
                    size=p.numpy().shape
                )

                # Assign those values to the trainable variable
                p.assign(p.transform_fn(unconstrain_vals))
        return None


    def optimize_params(
            self,
            # scipy_opts={
            #     'method': 'L-BFGS-B', 
            #     'options': {'maxiter': 5000}
            # },
            adam_learning_rate=0.01,
            nat_gradient_gamma=0.0001,
            num_opt_iter=5000,
            convergence_threshold=1e-6
        ):
        ''' Optimize hyperparameters of model.
        
        Parameters
        ----------
        
        Returns
        -------
        None
        '''
        # # Optimization step for hyperparameters
        # # Try the optimization but be ready for an exception
        # try:
        #     opt_res = gpflow.optimizers.Scipy().minimize(
        #         closure=self.training_loss_closure(
        #             data=(self.X, self.Y)
        #         ),
        #         variables=self.trainable_variables,
        #         **scipy_opts
        #     )

        #     if opt_res['success'] is False:
        #         print('Caution! Optimization of hyperparameters was not successful.')
            
        # except Exception as e:
        #     if self.verbose:
        #         print(f'Optimization error! Error: {e}')

        # Source: (https://gpflow.github.io/GPflow/develop/
        # notebooks/advanced/natural_gradients.html#Natural-gradients)

        # Stop Adam from optimizing the variational parameters
        gpflow.set_trainable(self.q_mu, False)
        gpflow.set_trainable(self.q_sqrt, False)

        # Create the optimize_tensors for VGP with natural gradients
        adam_opt = Adam(learning_rate=adam_learning_rate)
        natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=nat_gradient_gamma)

        # Compile loss closure based on training data
        compiled_loss = self.training_loss_closure(
            data=(self.X, self.Y),
            compile=True
        ) 


        @tf.function
        def split_optimization_step():
            adam_opt.minimize(
                compiled_loss,
                var_list=self.trainable_variables,
            )
            natgrad_opt.minimize(
                compiled_loss,
                var_list=[(self.q_mu, self.q_sqrt)]
            )

        loss_list = []
        # Now optimize the parameters
        for i in range(num_opt_iter):
            try:
                split_optimization_step()
            except tf.errors.InvalidArgumentError:
                print('Reached invalid step in optimization, returning previous step.')
                break
            if i % 5 == 0:
                cur_loss = self.training_loss((self.X, self.Y))
                loss_list.append(cur_loss)
                if i % 100 == 0 and self.verbose: print(f'Round {i} training loss: {cur_loss}')
                
            if len(loss_list) > 1 and loss_list[-2] - loss_list[-1] < convergence_threshold:
                if self.verbose: print(f'Optimization converged - stopping early (round {i})')
                break 
        
        if i == (num_opt_iter-1):
            print(f'Optimization not converged after {i+1} rounds')

        return None
    
    def random_restart_optimize(
            self,
            num_restart = 5,
            randomize_kwargs = {},
            optimize_kwargs = {}
    ):
        ''' Optimize hyperparameters a certain number of times to search.
        
        Parameters
        ----------

        Returns
        -------
        None

        '''  
        # Set initial log likelihood to track during restarts
        max_ll = -np.inf
        best_variables = {}

        for i in range(num_restart):
            if self.verbose: print(f'Random restart {i+1}')
            # Randomize parameters
            self.randomize_params(**randomize_kwargs)

            # Optimize parameters
            self.optimize_params(**optimize_kwargs)

            # Check if log likelihood is better
            cur_max_ll = self.maximum_log_likelihood_objective(data=(self.X, self.Y))
            if cur_max_ll > max_ll:
                max_ll = cur_max_ll
                best_variables = gpflow.utilities.deepcopy(
                    gpflow.utilities.parameter_dict(self)
                )
                if self.verbose: print('Found better parameters!')

        # Set trainable variables to the best found
        gpflow.utilities.multiple_assign(self, best_variables)

        return None

    def variance_explained(self):
        self.variance_explained = variance_contributions(
            self,
            k_names=self.kernel_name, 
            lik=self.likelihood.name
        )
    
    def calc_metric(self, metric='BIC'):
        assert metric == 'BIC', "Only BIC currently allowed."
        if metric == 'BIC':
            return  calc_bic(
                loglik=self.log_posterior_density(self.data), 
                n=self.X.shape[0], 
                k=len(self.trainable_parameters)
            )
    
    def plot_functions(self, x_idx, col_names, **kwargs):
        return gp_predict_fun(
            self, x_idx, col_names, 
            X=self.X, Y=self.Y,
            **kwargs)
    
    def plot_parts(self, x_idx, unit_idx, col_names, lik=None, **kwargs):
        if lik == None:
            lik = self.likelihood
        return pred_kernel_parts(
            self,
            x_idx=x_idx,
            unit_idx=unit_idx,
            col_names=col_names,
            lik=lik,
            **kwargs
        )

class VarGP(BaseGP):
    ''' Variational Gaussian process that inherits SVGP gpflow structure.

    Attributes
    ----------

    Methods
    -------
    
    '''
    def __init__(self, 
        X, Y,
        mean_function=gpflow.utilities.deepcopy(gpflow.mean_functions.Constant()),
        kernel=gpflow.utilities.deepcopy(gpflow.kernels.SquaredExponential()), 
        likelihood='gaussian',
        variational_priors=True, 
        verbose=False,
        **basegp_kwargs
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
            **basegp_kwargs
        )

        # Freeze inducing variables for this model type
        gpflow.utilities.set_trainable(
            self.inducing_variable, 
            False
        )

        # Now set likelihood based on argument
        if type(likelihood) is str:
            self.likelihood = gp_likelihood_crosswalk(likelihood)
        elif 'gpflow.likelihoods.' in str(type(likelihood)):
            self.likelihood = likelihood
        else:
            raise('Unknown likelihood requested. Either string or gpflow.likelihood allowed') 
        
        # Set variational priors if requested
        if variational_priors is True:
            self.q_mu.prior = tfd.Normal(loc=0., scale=100.)
            self.q_sqrt.prior = tfd.HalfNormal(scale=100.)


class SparseGP(BaseGP):
    ''' Sparse Gaussian process that inherits SVGP gpflow structure.

    Attributes
    ----------

    Methods
    -------
    
    '''
    def __init__(
            self, 
            X, 
            Y, 
            mean_function=gpflow.utilities.deepcopy(gpflow.mean_functions.Constant()),
            kernel=gpflow.utilities.deepcopy(gpflow.kernels.SquaredExponential()), 
            num_inducing_points=500, 
            train_inducing=True,
            random_points=True,
            random_seed=None,
            verbose=False,
            **basegp_kwargs
        ):
        # Set values
        self.X = X
        self.Y = Y
        self.mean_function = mean_function
        self.kernel = kernel
        self.verbose = verbose

        # Check inducing point size compared to training data
        # If more than we need then we set to dataset size and don't train
        if num_inducing_points > X.shape[0]:
            if verbose: 
                print(
                  f'Number of inducing points requested ({num_inducing_points}) ' 
                  f'greater than original data ({X.shape[0]})'
                )
            num_inducing_points = X.shape[0]
            train_inducing = False
        # assert num_inducing_points <= X.shape[0], (
        #           f'Number of inducing points requested ({num_inducing_points}) ' 
        #           f'greater than original data ({X.shape[0]})'
        # )

        # Fill in information for parent class
        super().__init__(
            X=X,
            Y=Y,
            kernel=kernel,
            verbose=verbose,
            q_mu=np.zeros(num_inducing_points)[:, None],
            q_sqrt=np.eye(num_inducing_points)[None, :, :],
            **basegp_kwargs
        )

        # Set inducing points
        if random_points is True:
            if random_seed is not None:
                np.random.seed(random_seed)
            obs_idx = np.random.choice(
                a=np.arange(X.shape[0]),
                size=num_inducing_points,
                replace=False
            )
        else:
            obs_idx = np.arange(num_inducing_points)

        self.inducing_variable = gpflow.inducing_variables.InducingPoints(
            X[obs_idx, :].copy()
        )
        gpflow.utilities.set_trainable(
            self.inducing_variable, 
            train_inducing
        )
         

class PenalizedGP(BaseGP):
    ''' Penalized Gaussian process via priors on variance hyperparameters.

    Attributes
    ----------

    Methods
    -------

    '''
    def __init__(
            self, 
            X, 
            Y, 
            mean_function=gpflow.utilities.deepcopy(gpflow.mean_functions.Constant()),
            kernel=gpflow.utilities.deepcopy(gpflow.kernels.SquaredExponential()), 
            penalization_factor=1.,
            verbose=False,
            **basegp_kwargs
    ):
        
        # Parent class
        super().__init__(
            X=X,
            Y=Y,
            mean_function=mean_function,
            kernel=kernel,
            verbose=verbose,
            **basegp_kwargs
        )

        # Set initial factor given
        self.set_penalization_factor(penalization_factor)

        # Set unit col as none for now
        self.unit_col = None
        self.penalization_search_results = None

    def set_penalization_factor(self, penalization_factor):
        self.penalization_factor = penalization_factor
        
        # Set prior on kernel variance terms
        if penalization_factor > 0:
            prior = tfd.Exponential(
                        rate=gpflow.utilities.to_default_float(1./penalization_factor)
                    )
        else:
            prior = None
        for key, val in gpflow.utilities.parameter_dict(self).items():
            if 'kernel' in key and 'variance' in key:
                val.prior = prior

    def penalization_search(
            self, 
            penalization_factor_list = [0., 1., 10., 100., 1000.], 
            k_fold = 3, 
            fit_best = True,
            max_jobs = -1,
            show_progress = True,
            parallel_object = None,
            randomization_options = {},
            optimization_options = {},
            random_seed = None
        ):
        
        # Split training data into k-folds
        folds = make_folds(self.X, self.unit_col, k_fold, random_seed)

        # Set random seed in randomization options if not defined
        if 'random_seed' not in randomization_options.keys():
            randomization_options['random_seed'] = random_seed

        # Figure out total combinations of fold x factor
        ff_idx = np.array(
            np.meshgrid(
                np.arange(len(penalization_factor_list)), 
                np.arange(len(folds))
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
                verbose=self.verbose
            )
            # temp_model.set_penalization_factor(pf)
            holdout_X = self.X[holdout_fold]
            holdout_Y = self.Y[holdout_fold]
            # temp_model.X = np.delete(self.X, holdout_fold, axis=0)
            # temp_model.Y = np.delete(self.Y, holdout_fold, axis=0)
            temp_model.randomize_params(**randomization_options)
            temp_model.optimize_params(**optimization_options)
            holdout = np.mean(
                temp_model.predict_log_density(
                    data=(holdout_X, holdout_Y)
                )
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
            # if show_progress:
            #     with tqdm_joblib(tqdm(desc="Penalization search", total=ff_idx.shape[0])) as progress_bar:
            #         parallel_results = parallel_object(delayed(parallel_fit)(
            #             pf=penalization_factor_list[i[0]],
            #             holdout_fold=folds[i[1]],
            #             holdout_index=i[1]
            #         ) for i in ff_idx)
            # else:
            parallel_results = parallel_object(delayed(parallel_fit)(
                pf=penalization_factor_list[i[0]],
                holdout_fold=folds[i[1]],
                holdout_index=i[1]
            ) for i in ff_idx) 

        else:
            if show_progress:
                with tqdm_joblib(tqdm(desc="Penalization search", total=ff_idx.shape[0])) as progress_bar:
                    parallel_results = Parallel(n_jobs=max_jobs)(delayed(parallel_fit)(
                        pf=penalization_factor_list[i[0]],
                        holdout_fold=folds[i[1]],
                        holdout_index=i[1]
                    ) for i in ff_idx)
            else:
                parallel_results = Parallel(n_jobs=max_jobs)(delayed(parallel_fit)(
                    pf=penalization_factor_list[i[0]],
                    holdout_fold=folds[i[1]],
                    holdout_index=i[1]
                ) for i in ff_idx)

        parallel_results = np.vstack(parallel_results)
        self.penalization_search_results = parallel_results

        # Find best penalization factor
        max_val = -np.inf
        max_factor = -np.inf
        for factor in penalization_factor_list:
            cur_val = parallel_results[parallel_results[:, 0] == factor, 2].mean()
            if cur_val > max_val:
                max_factor = factor
                max_val = cur_val
        best_factor = max_factor            

        if self.verbose: print(f'Best penalization factor found from search: {best_factor}')

        # Fit full model with best penalization value found
        if fit_best:
            self.set_penalization_factor(best_factor)
            self.randomize_params(**randomization_options)
            self.optimize_params(**optimization_options)
    
    def cut_kernel_components(
            self,
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
        if self is None:
            return self
        
        # Get variance components for each additive kernel part
        var_parts = find_variance_components(self.kernel, sum_reduce=False)
        var_flag = np.argwhere(var_parts >= var_cutoff).transpose().tolist()[0]

        # Copy model
        # new_model = gpflow.utilities.deepcopy(model)

        # Figure out which ones should be kept and sum if needed
        if len(var_flag) > 1:
            self.kernel = gpflow.kernels.Sum(
                [self.kernel.kernels[i] for i in var_flag]
            )
        else:
            self.kernel = self.kernel.kernels[var_flag[0]]

        # Also make sure we only keep base variances that remain
        # This will not be used post refactor of code base
        if hasattr(self, 'base_variances'):
            if self.base_variances is not None:
                self.base_variances = self.base_variances[var_flag]

        return None



class PSVGP(PenalizedGP, SparseGP, VarGP):
    ''' Combine all of the sub Gaussian process types into the main entry point.

    Attributes
    ----------

    Methods
    -------

    '''
    def __init__(
            self,
            X,
            Y,
            mean_function=gpflow.utilities.deepcopy(gpflow.mean_functions.Constant()),
            kernel=gpflow.utilities.deepcopy(gpflow.kernels.SquaredExponential()),
            verbose=False,
            penalized_options={},
            sparse_options={},
            variational_options={}
    ):
        
        super().__init__(
            X=X,
            Y=Y,
            mean_function=mean_function,
            kernel=kernel,
            verbose=verbose,
            **penalized_options,
            **sparse_options,
            **variational_options
        )