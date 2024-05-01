import numpy as np

from enum import Enum


# Type of the numbers used for calculation. Note that `real_type` can be used for both contructing scalars or for
# specifying `dtype` arguments. `real_dtype` should be used for comparing `dtype`s
real_type = np.float32
real_dtype = np.dtype(real_type)

# An approximative value of machine epsilon, which is useful in avoiding some numerical issues such as division
# by zero. Keras also uses this value, see https://github.com/tensorflow/tensorflow/blob/066e226b3ed6db054cdb5ed0ff2453b8c1ffb3f6/tensorflow/python/keras/backend_config.py#L24
epsilon = real_type(1e-7)


class PoolingMode(Enum):
    MAX = 1
    AVERAGE = 2
