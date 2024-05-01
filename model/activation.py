"""
@brief Contains components for building a neural network.
All 1-D vectors  of `n` elements are assumed to have shape = `(n, 1)` (a column vector).
"""


import common.vector as vec

import numpy as np
import numpy.typing as np_typing

from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    @abstractmethod
    def eval(self, z: np_typing.NDArray):
        """
        @param z The input vector.
        @return Vector of the evaluated function (activation, `a`).
        """
        pass
        
    @abstractmethod
    def jacobian(self, z: np_typing.NDArray, **kwargs):
        """
        @param z The input vector.
        @param kwargs Implementation defined extra arguments (e.g., to facilitate the calculation).
        @return A matrix of all the function's first-order derivatives with respect to `z`. For example,
        the 1st row would be (da(z1)/dz1, da(z1)/dz2, ..., da(z1)/dzn),
        the 2nd row would be (da(z2)/dz1, da(z2)/dz2, ..., da(z2)/dzn),
        and so on (all the way to da(zn)/dzn).
        """
        pass


class Identity(ActivationFunction):
    def eval(self, z):
        return z
    
    def jacobian(self, z, **kwargs):
        dadz = np.ones(z.shape, dtype=z.dtype)
        return vec.diag_from_vector_2d(dadz)


class Sigmoid(ActivationFunction):
    def eval(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def jacobian(self, z, **kwargs):
        # Sigmoid is element-wise independent, so its jacobian is simply a diagonal matrix
        a = kwargs['a'] if 'a' in kwargs else self.eval(z)
        dadz = a * (1 - a)
        return vec.diag_from_vector_2d(dadz)
    

class Softmax(ActivationFunction):
    def eval(self, z):
        # Improves numerical stability (does not change the result--will cancel out in the division)
        z = z - np.max(z, axis=-2, keepdims=True)
        e_z = np.exp(z)
        a = e_z / np.sum(e_z, axis=-2, keepdims=True)
        return a
    
    def jacobian(self, z, **kwargs):
        # For a derivation that is clean and avoids looping & branching, see
        # https://mattpetersen.github.io/softmax-with-cross-entropy
        a = kwargs['a'] if 'a' in kwargs else self.eval(z)
        diag_a = vec.diag_from_vector_2d(a)
        a_T = vec.transpose_2d(a)
        dadz = diag_a - a @ a_T
        return dadz


class Tanh(ActivationFunction):
    def eval(self, z):
        return np.tanh(z)

    def jacobian(self, z, **kwargs):
        a = kwargs['a'] if 'a' in kwargs else self.eval(z)
        dadz = 1 - np.square(a)
        return vec.diag_from_vector_2d(dadz)


class ReLU(ActivationFunction):
    def eval(self, z):
        return z * (z > 0)
    
    def jacobian(self, z, **kwargs):
        # Derivatives at 0 is implemented as 0. See "Numerical influence of ReLU'(0) on backpropagation",
        # https://hal.science/hal-03265059/file/Impact_of_ReLU_prime.pdf
        dadz = (z > 0).astype(z.dtype)
        return vec.diag_from_vector_2d(dadz) 
