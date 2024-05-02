"""
@brief Contains components for building a neural network.
All 1-D vectors  of `n` elements are assumed to have shape = `(n, 1)` (a column vector).
"""


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


class Reshape(LayerWrapper):
    """
    Reshapes inputs into another shape.
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
    
    def weighted_input(self, x):
        wrapped_x = x.reshape(self.wrapped.input_vector_shape)
        wrapped_z = self.wrapped.weighted_input(wrapped_x)
        return wrapped_z.reshape(self.output_vector_shape)
    
    def update_params(self, bias, weight):
        return self.wrapped.update_params(bias, weight)

    def derived_params(self, x, delta):
        wrapped_x = x.reshape(self.wrapped.input_vector_shape)
        wrapped_delta = delta.reshape(self.wrapped.output_vector_shape)
        return self.wrapped.derived_params(wrapped_x, wrapped_delta)

    def feedforward(self, x, **kwargs):
        """
        @param kwargs 'z': `weighted_input` of this layer (`x` will be ignored).
        """
        z = kwargs['z'] if 'z' in kwargs else self.weighted_input(x)
        return self.activation.eval(z)

    def backpropagate(self, x, delta):
        wrapped_x = x.reshape(self.wrapped.input_vector_shape)
        wrapped_delta = delta.reshape(self.wrapped.output_vector_shape)
        wrapped_dCdx = self.wrapped.backpropagate(wrapped_x, wrapped_delta)
        return wrapped_dCdx.reshape(self.input_vector_shape)
    
    def __str__(self):
        return f"reshape: {self.input_shape} -> {self.output_shape} ({self.wrapped})"
