# DEPRECATED
# Need to move this into model_classes.py

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.base import RegressionData
from gpflow.logdensities import multivariate_normal
from gpflow.utilities import add_likelihood_noise_cov

from .utilities import find_variance_components


class PGPR(gpflow.models.GPR):
    def __init__(
        self,
        data,
        kernel,
        mean_function=None,
        noise_variance=1.0,
        lam=1.0,
        base_variances=None,
        gam=1.0,
    ):
        super().__init__(data, kernel, mean_function, noise_variance)
        self.lam = lam
        self.gam = gam

        # Set base variances to ones if none given
        if base_variances is None:
            num_vars = len(find_variance_components(kernel, sum_reduce=False))
            self.base_variances = np.ones(shape=num_vars)
        else:
            self.base_variances = base_variances

    def set_lambda(self, new_lam):
        self.lam = new_lam

    def set_gamma(self, new_gam):
        self.gam = new_gam

    def log_marginal_likelihood(self, penalize=True) -> tf.Tensor:
        """
        Computes the log marginal likelihood.
        .. math::
            \log p(Y | \theta).
        """

        X, Y = self.data
        K = self.kernel(X)
        ks = add_likelihood_noise_cov(K, self.likelihood, X)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        pen_log_prob = tf.reduce_mean(log_prob) - tf.reduce_sum(
            (len(X))
            * self.lam
            * (1 / (self.base_variances) ** self.gam)
            * find_variance_components(self.kernel, sum_reduce=False)
        )
        # print("Original log prob:", tf.reduce_sum(log_prob))
        # print("Penalized log prob:", pen_log_prob)
        if penalize:
            return pen_log_prob
        else:
            return log_prob


class SVPGPR(gpflow.models.SVGP):
    def __init__(
        self,
        X,
        Y,
        kernel,
        inducing_variable=None,
        likelihood=gpflow.likelihoods.Gaussian(),
        mean_function=None,
        num_inducing_points=500,
        noise_variance=1.0,
        lam=1.0,
        base_variances=None,
        gam=1.0,
        **kwargs
    ):
        # Set inducing variables if needed
        fix_inducing_points = False
        if inducing_variable is None:
            if num_inducing_points >= len(X):
                fix_inducing_points = True
                inducing_variable = gpflow.inducing_variables.InducingPoints(X)
            else:
                sample_points = np.random.choice(len(X), num_inducing_points)
                inducing_variable = gpflow.inducing_variables.InducingPoints(
                    X[sample_points, :]
                )

        super().__init__(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=mean_function,
            **kwargs
        )
        self.lam = lam
        self.gam = gam
        self.data = (
            tf.convert_to_tensor(X, dtype=tf.float64),
            tf.convert_to_tensor(Y, dtype=tf.float64),
        )

        # Set inducing points
        if fix_inducing_points is True:
            gpflow.utilities.set_trainable(self.inducing_variable, False)

        # Set base variances to ones if none given
        self.base_variances = base_variances

    def set_lambda(self, new_lam):
        self.lam = new_lam

    def set_gamma(self, new_gam):
        self.gam = new_gam

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(
            X, full_cov=False, full_output_cov=False
        )
        var_exp = self.likelihood.variational_expectations(X, f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)

        if self.base_variances is None:
            pen_factor = tf.reduce_sum(
                (len(X))
                * self.lam
                * find_variance_components(self.kernel, sum_reduce=False)
            )
        else:
            pen_factor = tf.reduce_sum(
                (len(X))
                * self.lam
                * (1 / (self.base_variances) ** self.gam)
                * find_variance_components(self.kernel, sum_reduce=False)
            )

        return tf.reduce_sum(var_exp) * scale - kl - pen_factor
