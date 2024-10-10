import gpflow
import tensorflow as tf


class Lin(gpflow.kernels.Kernel):
    def __init__(self, active_dims=None, variance=1.0):
        super().__init__(active_dims=active_dims)

        # Set active dims if none provided
        if active_dims is None:
            active_dims = [0]

        self.variance = gpflow.Parameter(
            variance, transform=gpflow.utilities.positive()
        )
        # self.center = gpflow.Parameter(center)
        self.active_index = active_dims[0]

    def K(self, X, X2=None) -> tf.Tensor:
        if X.shape[1] > 1:
            X = tf.cast(
                tf.reshape(X[:, self.active_dims[0]], (-1, 1)), tf.float64
            )
        if X2 is None:
            return self.variance * tf.matmul(X, X, transpose_b=True)
        else:
            if X2.shape[1] > 1:
                X2 = tf.cast(
                    tf.reshape(X2[:, self.active_dims[0]], (-1, 1)), tf.float64
                )
            return self.variance * tf.matmul(X, X2, transpose_b=True)
            # return tf.tensordot(X * self.variance, X2, [[-1], [-1]])

    def K_diag(self, X) -> tf.Tensor:
        if len(X.shape) > 1 and X.shape[1] > 1:
            X = X[:, self.active_dims[0]]
        return self.variance * tf.cast(
            tf.reshape(tf.square(X), (-1,)), tf.float64
        )


class Poly(gpflow.kernels.Kernel):
    def __init__(
        self, active_dims=[0], variance=1.0, offset=1.0, degree=3
    ):  # , center=0.0):
        super().__init__(active_dims=active_dims)
        self.variance = gpflow.Parameter(
            variance, transform=gpflow.utilities.positive()
        )
        self.offset = gpflow.Parameter(
            offset, transform=gpflow.utilities.positive()
        )
        self.degree = degree
        self.active_index = active_dims[0]

    def K(self, X, X2=None) -> tf.Tensor:
        if X.shape[1] > 1:
            X = tf.cast(
                tf.reshape(X[:, self.active_dims[0]], (-1, 1)), tf.float64
            )
        if X2 is None:
            return (
                self.variance * tf.matmul(X, X, transpose_b=True) + self.offset
            ) ** self.degree
        else:
            if X2.shape[1] > 1:
                X2 = tf.cast(
                    tf.reshape(X2[:, self.active_dims[0]], (-1, 1)), tf.float64
                )
            return (
                self.variance * tf.matmul(X, X2, transpose_b=True)
                + self.offset
            ) ** self.degree
            # return tf.tensordot(X * self.variance, X2, [[-1], [-1]])

    def K_diag(self, X) -> tf.Tensor:
        if len(X.shape) > 1 and X.shape[1] > 1:
            X = X[:, self.active_dims[0]]
        return (
            self.variance
            * tf.cast(tf.reshape(tf.square(X), (-1,)), tf.float64)
            + self.offset
        ) ** self.degree


class Categorical(gpflow.kernels.Kernel):
    def __init__(self, active_dims=[0], variance=1.0):
        super().__init__(active_dims=active_dims)
        self.variance = gpflow.Parameter(
            variance, transform=gpflow.utilities.positive()
        )
        # self.rho = gpflow.Parameter(1.0, transform=positive())'
        self.active_index = active_dims[0]

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        # Select the single dimension if needed
        if X.shape[1] > 1:
            # print(f"X: {X}, shape(X): {(X.shape)}")
            X = tf.reshape(X[:, self.active_index], (-1, 1))
            X2 = tf.reshape(X2[:, self.active_index], (-1, 1))
        # matches = tf.cast(tf.equal(tf.cast(X, tf.int64),
        #                   tf.transpose(tf.cast(X2, tf.int64))),
        #                   tf.float64)
        # diagonals = tf.linalg.diag(self.variance/self.rho, matches.shape[0])
        # return matches
        return self.variance * tf.cast(
            tf.equal(
                # tf.cast(tf.reshape(X2, (1, -1)), tf.int64),
                # tf.cast(X, tf.int64)
                tf.cast(tf.round(tf.reshape(X2, (1, -1))), tf.int64),
                tf.cast(tf.round(X), tf.int64),
            ),
            tf.float64,
        )

    def K_diag(self, X):
        if len(X.shape) > 1 and X.shape[1] > 1:
            X = X[:, self.active_index]
        return self.variance * tf.cast(
            tf.reshape(tf.ones_like(X), (-1,)), tf.float64
        )


class Empty(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__(name="empty")
        self.active_dims = [0]
        self.variance = gpflow.Parameter(
            gpflow.utilities.to_default_float(1e-6),
            trainable=False
        )

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return tf.zeros_like(tf.linalg.matmul(X, tf.transpose(X2)))

    def K_diag(self, X):
        return tf.cast(tf.reshape(tf.zeros_like(X), (-1,)), tf.float64)
