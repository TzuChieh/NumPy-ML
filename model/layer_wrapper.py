"""
@brief Contains components for building a neural network.
All 1-D vectors  of `n` elements are assumed to have shape = `(n, 1)` (a column vector).
"""


import common as com
from model.layer import Layer

import numpy as np

from typing import Iterable


class LayerWrapper(Layer):
    """
    Wraps one layer and provides modified behavior.
    """
    def __init__(
        self,
        wrapped_layer: Layer):
        """
        @param wrapped_layer The layer being wrapped.
        """
        super().__init__()

        if wrapped_layer is self:
            raise ValueError("cannot wrap oneself")

        self._wrapped = wrapped_layer

    @property
    def wrapped(self) -> Layer:
        return self._wrapped


class FullyReshape(LayerWrapper):
    """
    Reshapes input and output into another shape.
    """
    def __init__(
        self,
        wrapped_layer: Layer,
        input_shape: Iterable[int]=None,
        output_shape: Iterable[int]=None):
        """
        @param input_shape Input dimensions. If `None`, will default to the shape of `wrapped_layer`.
        @param output_shape Output dimensions. If `None`, will default to the shape of `wrapped_layer`.
        """
        super().__init__(wrapped_layer)

        self._input_shape = np.array(input_shape) if input_shape is not None else wrapped_layer.input_shape
        self._output_shape = np.array(output_shape) if output_shape is not None else wrapped_layer.output_shape

        if self.input_shape.prod() != self.wrapped.input_shape.prod():
            raise ValueError(f"cannot convert input shape from {self.input_shape} to {self.wrapped.input_shape}")
        if self.output_shape.prod() != self.wrapped.output_shape.prod():
            raise ValueError(f"cannot convert output shape from {self.output_shape} to {self.wrapped.output_shape}")
        
        if (np.array_equal(self.input_shape, self.wrapped.input_shape) or 
            np.array_equal(self.output_shape, self.wrapped.output_shape)):
            print("A fully reshape layer wrapper effectively does nothing, consider removing it.")

    @property
    def bias(self):
        return self.wrapped.bias
    
    @property
    def weight(self):
        return self.wrapped.weight
    
    @property
    def activation(self):
        return self.wrapped.activation
    
    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape
    
    def update_params(self, bias, weight):
        assert self.is_trainable

        return self.wrapped.update_params(bias, weight)
    
    def weighted_input(self, x, cache):
        wrapped_x = x.reshape(self.wrapped.input_vector_shape)
        wrapped_z = self.wrapped.weighted_input(wrapped_x, cache)
        return wrapped_z.reshape(self.output_vector_shape)

    def derived_params(self, x, delta, cache):
        wrapped_x = x.reshape(self.wrapped.input_vector_shape)
        wrapped_delta = delta.reshape(self.wrapped.output_vector_shape)
        return self.wrapped.derived_params(wrapped_x, wrapped_delta, cache)

    def backpropagate(self, x, delta, cache):
        wrapped_x = x.reshape(self.wrapped.input_vector_shape)
        wrapped_delta = delta.reshape(self.wrapped.output_vector_shape)
        wrapped_dCdx = self.wrapped.backpropagate(wrapped_x, wrapped_delta, cache)
        return wrapped_dCdx.reshape(self.input_vector_shape)
    
    def __str__(self):
        return f"fully reshape: {self.input_shape} -> {self.output_shape} ({self.wrapped})"
