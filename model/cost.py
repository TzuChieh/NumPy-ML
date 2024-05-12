"""
@brief Contains components for building a neural network.
All 1-D vectors  of `n` elements are assumed to have shape = `(n, 1)` (a column vector).
"""


import common as com
import common.vector as vec

import numpy as np

from abc import ABC, abstractmethod


class CostFunction(ABC):
    """
    Abstraction for a cost function. The interface accepts inputs with higher dimensional components.
    In such case, the calculation will simply broadcast to those dimensions.
    """
    @abstractmethod
    def eval(self, a, y):
        """
        @param a The output vector.
        @param y The desired output vector.
        @return Vector of cost values.
        """
        pass

    @abstractmethod
    def derived_eval(self, a, y):
        """
        This method would have return a jacobian like `ActivationFunction.jacobian()`. However, currently all
        implementations are element-wise independent so we stick with returning a vector.
        @param a The output vector.
        @param y The desired output vector.
        @return Vector of derived cost values.
        """
        pass


class Quadratic(CostFunction):
    def eval(self, a, y):
        """
        Basically computing the per-element squared error between activations `a` and desired output `y`.
        The 0.5 multiplier is to make its derived form cleaner (for convenience).
        """
        assert vec.is_vector_2d(a)
        assert vec.is_vector_2d(y)

        C = com.REAL_TYPE(0.5) * np.linalg.norm(a - y, axis=-2, keepdims=True) ** 2
        return C
    
    def derived_eval(self, a, y):
        dCda = a - y
        return dCda
    

class CrossEntropy(CostFunction):
    def eval(self, a, y):
        """
        Calculates cross-entropy error on a per-element basis.
        """
        assert vec.is_vector_2d(a)
        assert vec.is_vector_2d(y)

        a = self._adapt_a(a)
        C = np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a), axis=-2, keepdims=True)
        return C

    def derived_eval(self, a, y):
        a = self._adapt_a(a)
        dCda = (a - y) / (a * (1 - a))
        return dCda
    
    def _adapt_a(self, a):
        """
        Slightly modify `a` to prevent division by 0 and NaN/Inf in various cost function operations (e.g.,
        the `(1 - y) * log(1 - a)` term can be NaN/Inf).
        """
        # We modify `a` similar to Keras: https://github.com/tensorflow/tensorflow/blob/066e226b3ed6db054cdb5ed0ff2453b8c1ffb3f6/tensorflow/python/keras/backend.py#L5046.
        # `np.nan_to_num()` could be an alternative solution, but it is more intrusive and needs to be tailored
        # for each method.
        return np.clip(a, com.EPSILON, 1 - com.EPSILON)
