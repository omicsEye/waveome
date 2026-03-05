import gpflow
import tensorflow as tf
import functools
from .utilities import set_precision, get_precision

# Monkey patch GPflow to automatically cast inputs to the default float type.
# This ensures float32/float64 compatibility even when users pass numpy arrays 
# of a different precision to the prediction methods of ANY GPflow model.

_original_predict_f = gpflow.models.GPModel.predict_f
_original_predict_y = gpflow.models.GPModel.predict_y

@functools.wraps(_original_predict_f)
def _casted_predict_f(self, Xnew, full_cov=False, full_output_cov=False):
    Xnew = tf.cast(Xnew, gpflow.default_float())
    return _original_predict_f(self, Xnew, full_cov, full_output_cov)

@functools.wraps(_original_predict_y)
def _casted_predict_y(self, Xnew, full_cov=False, full_output_cov=False):
    Xnew = tf.cast(Xnew, gpflow.default_float())
    return _original_predict_y(self, Xnew, full_cov, full_output_cov)

gpflow.models.GPModel.predict_f = _casted_predict_f
gpflow.models.GPModel.predict_y = _casted_predict_y