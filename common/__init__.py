import numpy as np

from enum import Enum


# Disabling division by 0 error. 0/0 division error can also be ignored by setting `invalid='ignore'`, but we
# choose to enable it so we can get notified for potential numerical issues. 
np.seterr(divide='ignore')


class EParamInit(Enum):
    GAUSSIAN = 1
    """
    Random values that form a standard normal distribution.
    """
    LECUN = 2
    """
    Scaled version of `GAUSSIAN`. In theory works better for sigmoid and tanh neurons.
    """
    XAVIER = 3
    """
    Considers both the input and output sizes of a layer, balancing the magnitude of computed signal for forward
    and backward passes. Also works better for sigmoid and tanh neurons, in theory.
    """
    KAIMING_HE = 4
    """
    An initialization method designed for ReLU and its variants. Also known as He initialization.
    """


class EPooling(Enum):
    MAX = 1
    """
    Picking the maximum element.
    """
    AVERAGE = 2
    """
    Compute a mean across all elements.
    """


REAL_TYPE = np.float32
"""
Type of the numbers used for calculation. Note that `REAL_TYPE` can be used for both contructing scalars or for
specifying `dtype` arguments. `REAL_DTYPE` should be used for comparing `dtype`s.
"""

REAL_DTYPE = np.dtype(REAL_TYPE)
"""
@see `REAL_TYPE`
"""

EPSILON = REAL_TYPE(1e-7)
"""
An approximative value of machine EPSILON, which is useful in avoiding some numerical issues such as division
by zero. Keras also uses this value, see https://github.com/tensorflow/tensorflow/blob/066e226b3ed6db054cdb5ed0ff2453b8c1ffb3f6/tensorflow/python/keras/backend_config.py#L24.
"""
