"""
@brief Contains parameter initializers.
"""

import numpy as np
import numpy.typing as np_type

from abc import ABC, abstractmethod


class Initializer(ABC):
    """
    Abstraction for a parameter initializer. The interface accepts inputs with higher dimensional components.
    In such case, the calculation will simply broadcast to those dimensions.
    """
    @abstractmethod
    def init(self, p: np_type.NDArray, fan_in=None, fan_out=None):
        """
        Initializes the parameter array `p` in-place. See https://stackoverflow.com/questions/42670274/how-to-calculate-fan-in-and-fan-out-in-xavier-initialization-for-neural-networks
        for more information on how to determine `fan_in` and `fan_out` for a layer.
        @param p The parameter array.
        @param fan_in How many inputs will contribute to a single output.
        @param fan_out How many outputs will contribute to a single input.
        """
        pass


class Constant(Initializer):
    """
    Filling with a chosen constant value.
    """
    def __init__(self, value):
        """
        @param value The value to use as the constant.
        """
        super().__init__()

        self._value = value

    def init(self, p, fan_in=None, fan_out=None):
        scalar = p.dtype.type(self._value)
        p.fill(scalar)


class Zeros(Constant):
    """
    Filling with zeros.
    """
    def __init__(self):
        super().__init__(0)


class Ones(Constant):
    """
    Filling with ones.
    """
    def __init__(self):
        super().__init__(1)


class Gaussian(Initializer):
    """
    Random values that form a standard normal distribution.
    """
    def __init__(self):
        super().__init__()

        self._rng = np.random.default_rng()

    def init(self, p, fan_in=None, fan_out=None):
        self._rng.standard_normal(p.shape, dtype=p.dtype, out=p)
    

class LeCun(Initializer):
    """
    Scaled version of `GAUSSIAN`. In theory works better for sigmoid and tanh neurons.
    """
    def __init__(self):
        super().__init__()

        self._rng = np.random.default_rng()

    def init(self, p, fan_in=None, fan_out=None):
        assert fan_in is not None

        self._rng.standard_normal(p.shape, dtype=p.dtype, out=p)
        p *= np.sqrt(1 / fan_in, dtype=p.dtype)


class Xavier(Initializer):
    """
    Considers both the input and output sizes of a layer, balancing the magnitude of computed signal for forward
    and backward passes. Also works better for sigmoid and tanh neurons, in theory.
    """
    def __init__(self):
        super().__init__()

        self._rng = np.random.default_rng()

    def init(self, p, fan_in=None, fan_out=None):
        assert fan_in is not None
        assert fan_out is not None

        self._rng.standard_normal(p.shape, dtype=p.dtype, out=p)
        p *= np.sqrt(2 / (fan_in + fan_out), dtype=p.dtype)


class KaimingHe(Initializer):
    """
    An initialization method designed for ReLU and its variants. Also known as He initialization.
    """
    def __init__(self):
        super().__init__()

        self._rng = np.random.default_rng()

    def init(self, p, fan_in=None, fan_out=None):
        assert fan_in is not None

        self._rng.standard_normal(p.shape, dtype=p.dtype, out=p)
        p *= np.sqrt(2 / fan_in, dtype=p.dtype)
