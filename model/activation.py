"""
@brief Contains components for building a neural network.
All 1-D vectors  of `n` elements are assumed to have shape = `(n, 1)` (a column vector).
"""


import common as com
import common.vector as vec

import numpy as np
import numpy.typing as np_typing

from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    """
    Abstraction for an activation function. The interface accepts inputs with higher dimensional components.
    In such case, the calculation will simply broadcast to those dimensions.
    """
    @abstractmethod
    def eval(self, z: np_typing.NDArray):
        """
        @param z The input vector.
        @return Activation `a`, a vector of the evaluated function.
        """
        pass
        
    def jacobian(self, z: np_typing.NDArray, **kwargs) -> np_typing.NDArray:
        """
        @param z The input vector.
        @param kwargs Implementation defined extra arguments (e.g., to facilitate the calculation).
        @return Derivative of the activation `da/dz`, a matrix of all the function's first-order derivatives
        with respect to `z`. For example,
        the 1st row would be (da(z1)/dz1, da(z2)/dz1, ..., da(zn)/dz1),
        the 2nd row would be (da(z1)/dz2, da(z2)/dz2, ..., da(zn)/dz2),
        the nth row would be (da(z1)/dzn, da(z2)/dzn, ..., da(zn)/dzn).
        Notice that there are two conventions at play in the field of multivariable calculus: the convention we use
        allow chain rule to be performed as a left multiplication of jacobian matrices, e.g., dC/dz = da/dz @ dC/da.
        The other convention essentially makes every component in the equation transposed.
        @see https://math.stackexchange.com/questions/3208939/transpose-of-product-of-matrices
        """
        assert vec.is_vector_2d(z)

        diag_j = self.diagonal_jacobian(z, **kwargs)
        assert diag_j is not None, "`jacobian()` must be implemented if jacobian is not diagonal"

        j = vec.diag_from_vector_2d(diag_j)
        return j
    
    def diagonal_jacobian(self, z: np_typing.NDArray, **kwargs) -> np_typing.NDArray:
        """
        Diagonal version of `jacobian()` in the implicit form. Jacobian matrix of a multivariate function is simply
        a diagonal matrix if it is element-wise independent.
        @param z The input vector.
        @return The diagonal elements of a diagonal jacobian, in the form of a vector. Broadcast accordingly to
        higher dimensions of `z`. `None` if the jacobian is not diagonal.
        @see `jacobian()`
        """
        return None

    def jacobian_mul(
        self,
        z: np_typing.NDArray,
        right: np_typing.NDArray=None,
        left: np_typing.NDArray=None,
        **kwargs):
        """
        Perform matrix multiplication with the jacobian. This method can potentially be more efficient as the jacobian
        may not need to be explicitly obtained via `jacobian()`.
        @param z The input vector.
        @param right The matrix to multiply on the right side of the jacobian.
        @param left The matrix to multiply on the left side of the jacobian.
        """
        assert vec.is_vector_2d(z)

        diag_j = self.diagonal_jacobian(z, **kwargs)
        j = self.jacobian(z, **kwargs) if diag_j is None else None

        # Only one of `diag_j` and `j` can exist
        assert (diag_j is not None and j is None) or (diag_j is None and j is not None)

        if diag_j is not None:
            assert vec.is_vector_2d(diag_j)

            # As elementwise multiplications
            result = diag_j
            if right is not None:
                result = result * right
            if left is not None:
                result = left * result
        else:
            assert j is not None

            # As matrix multiplications
            result = j
            if right is not None:
                result = result @ right
            if left is not None:
                result = left @ result

        return result


class Identity(ActivationFunction):
    def eval(self, z):
        return z
    
    def diagonal_jacobian(self, z, **kwargs):
        dadz = np.ones(z.shape, dtype=z.dtype)
        return dadz


class Sigmoid(ActivationFunction):
    def eval(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def diagonal_jacobian(self, z, **kwargs):
        a = kwargs['a'] if 'a' in kwargs else self.eval(z)
        dadz = a * (1 - a)
        return dadz
    

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
    """
    Hyperbolic tangent (tanh).
    """
    def eval(self, z):
        return np.tanh(z)

    def diagonal_jacobian(self, z, **kwargs):
        a = kwargs['a'] if 'a' in kwargs else self.eval(z)
        dadz = 1 - np.square(a)
        return dadz


class ReLU(ActivationFunction):
    """
    Rectified linear unit (ReLU).
    """
    def eval(self, z):
        return z * (z > 0)
    
    def diagonal_jacobian(self, z, **kwargs):
        # Derivatives at 0 is implemented as 0. See "Numerical influence of ReLU'(0) on backpropagation",
        # https://hal.science/hal-03265059/file/Impact_of_ReLU_prime.pdf
        dadz = (z > 0).astype(z.dtype)
        return dadz

class LeakyReLU(ActivationFunction):
    """
    A "leaky" version of `ReLU`, where the gradient is no longer 0 when input is negative. This function is
    designed to ameliorate the vanishing gradient problem.
    """
    def __init__(self, negative_slope=0.02):
        super().__init__()

        self._alpha = com.REAL_TYPE(negative_slope)

    def eval(self, z):
        return z * (z > 0) + self._alpha * z * (z <= 0)
    
    def diagonal_jacobian(self, z, **kwargs):
        # Derivatives at 0 is implemented as `self._alpha`
        dadz = (z > 0) + self._alpha * (z <= 0)
        return dadz
