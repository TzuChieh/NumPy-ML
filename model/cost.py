"""
@brief Contains components for building a neural network.
All 1-D vectors  of `n` elements are assumed to have shape = `(n, 1)` (a column vector).
"""


import common as com

import numpy as np

from abc import ABC, abstractmethod


class CostFunction(ABC):
    @abstractmethod
    def eval(self, a, y):
        """
        @param a The input vector.
        @param y The desired input vector.
        @return A vector of cost values.
        """
        pass

    @abstractmethod
    def derived_eval(self, a, y):
        """
        This method would have return a jacobian like `ActivationFunction.jacobian()`. However, currently all
        implementations are element-wise independent so we stick with returning a vector.
        @param a The input vector.
        @param y The desired input vector.
        @return A vector of derived cost values.
        """
        pass


class Quadratic(CostFunction):
    def eval(self, a, y):
        """
        Basically computing the squared error between activations `a` and desired output `y`. The 0.5 multiplier
        is to make its derived form cleaner (for convenience).
        """
        C = com.REAL_TYPE(0.5) * np.linalg.norm(a - y) ** 2
        return C
    
    def derived_eval(self, a, y):
        dCda = a - y
        return dCda
    

class CrossEntropy(CostFunction):
    def eval(self, a, y):
        a = self._adapt_a(a)
        C = np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a))
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
