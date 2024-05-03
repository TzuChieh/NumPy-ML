"""
@brief Contains components for building a neural network.
All 1-D vectors  of `n` elements are assumed to have shape = `(n, 1)` (a column vector).
"""


import common as com
import common.vector as vec
from model.activation import ActivationFunction, Sigmoid, Identity

import numpy as np
import numpy.typing as np_type

import typing
from abc import ABC, abstractmethod


"""
Temporary storage type for implementation defined extra arguments (e.g., to facilitate the calculation). A single
cache instance can be used across multiple layers by first using the layer instance as key.
@see `create_layer_cache()`
"""
type LayerCache = typing.Dict[object, typing.Dict]


class Layer(ABC):
    """
    Abstraction for a single layer in a neural network. Parameters should be of the defined shape (e.g., a vector)
    in the lowest dimensions, and be broadcastable to higher dimensions.
    """
    def __init__(self):
        super().__init__()

        self._is_trainable = True

    @property
    @abstractmethod
    def bias(self) -> np_type.NDArray:
        """
        @return Bias parameters. Empty array if there is no bias term.
        """
        pass

    @property
    @abstractmethod
    def weight(self) -> np_type.NDArray:
        """
        @return Weight parameters. Empty array if there is no weight term.
        """
        pass

    @property
    @abstractmethod
    def activation(self) -> ActivationFunction:
        """
        @return The activation function.
        """
        pass

    @property
    @abstractmethod
    def input_shape(self) -> np_type.NDArray:
        """
        @return The dimensions of input in (..., number of channels, height, width).
        """
        pass

    @property
    @abstractmethod
    def output_shape(self) -> np_type.NDArray:
        """
        @return The dimensions of output in (..., number of channels, height, width).
        """
        pass

    @abstractmethod
    def update_params(self, bias: np_type.NDArray, weight: np_type.NDArray):
        """
        Update layer parameters.
        @param bias The new bias parameters.
        @param weight The new weight parameters.
        """
        pass

    @abstractmethod
    def weighted_input(self, x: np_type.NDArray, cache: LayerCache) -> np_type.NDArray:
        """
        @param x The input vector.
        @param cache Implementation defined extra arguments (e.g., to facilitate the calculation).
        @return The weighted input vector (z).
        """
        pass

    @abstractmethod
    def derived_params(self, x: np_type.NDArray, delta: np_type.NDArray, cache: LayerCache):
        """
        @param x The input vector.
        @param delta The error vector (dCdz).
        @param cache Implementation defined extra arguments (e.g., to facilitate the calculation).
        @return Gradient of `bias` and `weight` in the form `(del_b, del_w)`. A derived term could be an empty array
        if there is no such parameter (e.g., a layer with empty bias could return `(empty array, del_w)`).
        """
        pass

    @abstractmethod
    def backpropagate(self, x: np_type.NDArray, delta: np_type.NDArray, cache: LayerCache) -> np_type.NDArray:
        """
        @param x The input vector.
        @param delta The error vector (dCdz).
        @param cache Implementation defined extra arguments (e.g., to facilitate the calculation).
        @return The error vector (dCdx); or equivalently, "dCda'" (where "a'" is the activation from previous layer).
        """
        pass

    @property
    def input_vector_shape(self) -> np_type.NDArray:
        """
        @return A compatible `input_shape` in the form of vector.
        """
        return (*self.input_shape[:-2], self.input_shape[-2:].prod(), 1)

    @property
    def output_vector_shape(self) -> np_type.NDArray:
        """
        @return A compatible `output_shape` in the form of vector.
        """
        return (*self.output_shape[:-2], self.output_shape[-2:].prod(), 1)
    
    @property
    def num_params(self) -> int:
        """
        @return Number of learnable parameters.
        """
        return self.bias.size + self.weight.size
    
    @property
    def is_trainable(self) -> bool:
        """
        @return Whether the layer is trainable or not. By default, the layer is trainable. Trainability can be
        updated by calling `freeze()` and `unfreeze()`.
        """
        return self._is_trainable

    def feedforward(self, x, cache: LayerCache=None):
        """
        @param x The input vector.
        @param cache Implementation defined extra arguments (e.g., to facilitate the calculation).
        @return Activation vector of the layer.
        """
        z = self.try_get_from_cache(cache, 'z')
        if z is None:
            z = self.weighted_input(x, cache)

        return self.activation.eval(z)

    def init_normal_params(self):
        """
        Randomly set initial parameters. The random values forms a standard normal distribution.
        """
        rng = np.random.default_rng()
        b = rng.standard_normal(self.bias.shape, dtype=com.REAL_TYPE)
        w = rng.standard_normal(self.weight.shape, dtype=com.REAL_TYPE)
        self.update_params(b, w)

    def init_scaled_normal_params(self):
        """
        Scaled (weights) version of `init_normal_params()`. In theory works better for sigmoid and tanh neurons.
        """
        rng = np.random.default_rng()
        nx = self.input_vector_shape[-2]
        b = rng.standard_normal(self.bias.shape, dtype=com.REAL_TYPE)
        w = rng.standard_normal(self.weight.shape, dtype=com.REAL_TYPE) / np.sqrt(nx, dtype=com.REAL_TYPE)
        self.update_params(b, w)

    def freeze(self):
        """
        Update the layer such that it is untrainable.
        """
        self._is_trainable = False

    def unfreeze(self):
        """
        Update the layer such that it is trainable.
        """
        self._is_trainable = True

    def try_cache(self, cache: LayerCache, name: str, value):
        if cache is None:
            return
        
        if self not in cache:
            cache[self] = {}
        
        cache[self][name] = value

    def try_get_from_cache(self, cache: LayerCache, name: str):
        if cache is None or self not in cache or name not in cache[self]:
            return None
        
        return cache[self][name]


class FullyConnected(Layer):
    """
    A fully connected network layer.
    """
    def __init__(
        self, 
        input_shape: typing.Iterable[int],
        output_shape: typing.Iterable[int], 
        activation: ActivationFunction=Sigmoid()):
        """
        @param input_shape Input dimensions, in (number of channels, height, width).
        @param output_shape Output dimensions, in (number of channels, height, width).
        """
        super().__init__()
        self._input_shape = np.array(input_shape)
        self._output_shape = np.array(output_shape)
        self._activation = activation

        assert input_shape[-3] == output_shape[-3], (
            f"number of input channels {input_shape[-3]} must match number of output channels {output_shape[-3]}")
        nc = self.input_shape[-3]
        ny = self.output_vector_shape[-2]
        nx = self.input_vector_shape[-2]
        self._bias = np.zeros((nc, ny, 1), dtype=com.REAL_TYPE)
        self._weight = np.zeros((nc, ny, nx), dtype=com.REAL_TYPE)

        self.init_scaled_normal_params()

    @property
    def bias(self):
        return self._bias
    
    @property
    def weight(self):
        return self._weight
    
    @property
    def activation(self):
        return self._activation
    
    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape
    
    def update_params(self, bias, weight):
        assert self.is_trainable
        assert self._bias.dtype == com.REAL_DTYPE, f"{self._bias.dtype}"
        assert self._weight.dtype == com.REAL_DTYPE, f"{self._weight.dtype}"
        assert np.array_equal(self._bias.shape, bias.shape), f"shapes: {self._bias.shape}, {bias.shape}"
        assert np.array_equal(self._weight.shape, weight.shape), f"shapes: {self._weight.shape}, {weight.shape}"

        self._bias = bias
        self._weight = weight

    def weighted_input(self, x, cache):
        z = self.try_get_from_cache(cache, 'z')
        if z is not None:
            return z

        b = self._bias
        w = self._weight
        z = w @ x + b

        self.try_cache(cache, 'z', z)
        return z

    def derived_params(self, x, delta, cache):
        del_b = np.copy(delta)
        x_T = vec.transpose_2d(x)
        del_w = delta @ x_T
        return (del_b, del_w)

    def backpropagate(self, x, delta, cache):
        assert delta.dtype == com.REAL_DTYPE, f"{delta.dtype}"

        w_T = vec.transpose_2d(self._weight)
        dCdx = w_T @ delta
        return dCdx
    
    def __str__(self):
        return f"fully connected: {self.input_shape} -> {self.output_shape} ({self.num_params})"


class Convolution(Layer):
    """
    A convolutional network layer. Note that the definition of convolution is correlating a kernel in reversed
    order, but in the field of ML I noticed that almost all the sources that I could find uses correlation instead.
    So here follows the same convention, namely, using correlation in the forward pass.
    """
    def __init__(
        self, 
        input_shape: typing.Iterable[int],
        kernel_shape: typing.Iterable[int],
        output_features: int,
        stride_shape: typing.Iterable[int]=(1,), 
        activation: ActivationFunction=Sigmoid()):
        """
        @param kernel_shape Kernel dimensions, in (height, width). Will automatically infer the number of channels
        of the kernel from `input_shape`.
        @param stride_shape Stride dimensions. The shape must be broadcastable to `kernel_shape`.
        """
        super().__init__()
        
        assert len(kernel_shape) == 2

        input_channels = input_shape[-3]
        stride_shape = (1, *np.broadcast_to(stride_shape, len(kernel_shape)))
        kernel_shape = (input_channels, *kernel_shape)
        correlated_shape = vec.correlate_shape(input_shape, kernel_shape, stride_shape)
        assert correlated_shape[-3] == 1

        self._input_shape = np.array(input_shape)
        self._output_shape = np.array((*correlated_shape[:-3], output_features, *correlated_shape[-2:]))
        self._kernel_shape = np.array(kernel_shape)
        self._stride_shape = np.array(stride_shape)
        self._activation = activation
        self._bias = np.zeros((output_features, 1, 1, 1), dtype=com.REAL_TYPE)
        self._weight = np.zeros((output_features, *kernel_shape), dtype=com.REAL_TYPE)

        self.init_scaled_normal_params()

    @property
    def bias(self):
        return self._bias
    
    @property
    def weight(self):
        return self._weight
    
    @property
    def activation(self):
        return self._activation
    
    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape
    
    def update_params(self, bias, weight):
        assert self.is_trainable

        self._bias = bias
        self._weight = weight

    def weighted_input(self, x, cache):
        z = self.try_get_from_cache(cache, 'z')
        if z is not None:
            return z

        x = x.reshape(self.input_shape)

        z = np.zeros(self.output_shape, dtype=x.dtype)
        for output_channel in range(self.output_shape[-3]):
            b = self._bias[output_channel]
            k = self._weight[output_channel]
            z[output_channel] = vec.correlate(x, k, self._stride_shape) + b
        z = z.reshape(self.output_vector_shape)

        self.try_cache(cache, 'z', z)
        return z

    def derived_params(self, x, delta, cache):
        x = x.reshape(self.input_shape)
        delta = delta.reshape(self.output_shape)

        # Backpropagation for bias is per-feature summation of gradient
        sum_axes = (-2, -1)
        del_b = delta.sum(axis=sum_axes, keepdims=True, dtype=delta.dtype).reshape(self.bias.shape)

        # Backpropagation is equivalent to a stride-1 correlation of input with a dilated gradient
        dilated_delta = vec.dilate(delta, self._stride_shape)
        del_w = vec.zeros_from(self.weight)
        for output_channel in range(self.output_shape[-3]):
            d = dilated_delta[np.newaxis, output_channel, ...]
            del_w[output_channel] = vec.correlate(x, d)

        return (del_b, del_w)

    def backpropagate(self, x, delta, cache):
        assert delta.dtype == com.REAL_DTYPE, f"{delta.dtype}"

        delta = delta.reshape(self.output_shape)

        # Backpropagation is equivalent to a stride-1 full correlation of a dilated (and padded) gradient with
        # a reversed kernel
        flip_axes = tuple(di for di in range(-len(self._kernel_shape), 0))
        reversed_k = np.flip(self._weight, axis=flip_axes)
        pad_shape = np.subtract(self._kernel_shape, 1)
        dilated_delta = vec.dilate(delta, self._stride_shape, pad_shape=pad_shape)
        dCdx = np.zeros(self.input_shape, dtype=delta.dtype)
        for output_channel in range(self.output_shape[-3]):
            k = reversed_k[output_channel]
            d = dilated_delta[np.newaxis, output_channel, ...]
            dCdx += vec.correlate(d, k)

        return dCdx.reshape(self.input_vector_shape)
    
    def __str__(self):
        kernel_info = "x".join(str(ks) for ks in self._kernel_shape)
        return f"{kernel_info} convolution: {self.input_shape} -> {self.output_shape} ({self.num_params})"


class Pool(Layer):
    """
    A max pooling layer.
    @note Currently the implementation of pooling is fairly slow--using manual iteration on the Python side for
    each pool window. For a simple network (~4 layers) with just one max pooling (2 -> 1 feature) slows down each
    epoch by a factor of ~2. TL;DR: a faster `backpropagate()` implementation is much appreciated.
    """
    def __init__(
        self, 
        input_shape: typing.Iterable[int],
        kernel_shape: typing.Iterable[int],
        mode: com.PoolingMode=com.PoolingMode.MAX,
        stride_shape: typing.Iterable[int]=None):
        """
        @param kernel_shape Pool dimensions.
        @param stride_shape Stride dimensions. The shape must be broadcastable to `kernel_shape` or being `None`.
        If `None`, will default to `kernel_shape`.
        """
        super().__init__()
        
        stride_shape = kernel_shape if stride_shape is None else np.broadcast_to(stride_shape, len(kernel_shape))

        self._input_shape = np.array(input_shape)
        self._output_shape = np.array(vec.pool_shape(input_shape, kernel_shape, stride_shape))
        self._kernel_shape = np.array(kernel_shape)
        self._rcp_num_kernel_elements = np.reciprocal(self._kernel_shape.prod(), dtype=com.REAL_TYPE)
        self._mode = mode
        self._stride_shape = np.array(stride_shape)

    @property
    def bias(self):
        return np.array([])
    
    @property
    def weight(self):
        return np.array([])
    
    @property
    def activation(self):
        return Identity()
    
    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape
    
    def update_params(self, bias, weight):
        pass

    def weighted_input(self, x, cache):
        z = self.try_get_from_cache(cache, 'z')
        if z is not None:
            return z

        x = x.reshape(self.input_shape)
        z = vec.pool(x, self._kernel_shape, self._stride_shape, self._mode)
        assert np.array_equal(z.shape, self.output_shape), f"shapes: {z.shape}, {self.output_shape}"
        z = z.reshape(self.output_vector_shape)

        self.try_cache(cache, 'z', z)
        return z

    def derived_params(self, x, delta, cache):
        return (np.array([]), np.array([]))

    def backpropagate(self, x, delta, cache):
        assert delta.dtype == com.REAL_DTYPE, f"{delta.dtype}"

        x = x.reshape(self.input_shape)
        delta = delta.reshape(self.output_shape)
        x_pool_view = vec.sliding_window_view(x, self._kernel_shape, self._stride_shape)

        dCdx = np.zeros(self.input_shape, dtype=delta.dtype)
        dCdx_pool_view = vec.sliding_window_view(dCdx, self._kernel_shape, self._stride_shape, is_writeable=True)
        for pool_idx in np.ndindex(*self.output_shape):
            x_pool = x_pool_view[pool_idx]
            dCdx_pool = dCdx_pool_view[pool_idx]

            # Lots of indexing here, make sure we are getting expected shapes
            assert np.array_equal(x_pool.shape, self._kernel_shape), f"shapes: {x_pool.shape}, {self._kernel_shape}"
            assert np.array_equal(x_pool.shape, dCdx_pool.shape), f"shapes: {x_pool.shape}, {dCdx_pool.shape}"
            
            match self._mode:
                # Gradient only propagate to the max element (think of an imaginary weight of 1, non-max element
                # has 0 weight)
                case com.PoolingMode.MAX:
                    dCdx_pool.flat[x_pool.argmax()] += delta[pool_idx]
                # Similar to the case of max pooling, average pooling is equivalent to an imaginary weight of
                # the reciprocal of number of pool elements
                case com.PoolingMode.AVERAGE:
                    dCdx_pool += delta[pool_idx] * self._rcp_num_kernel_elements
                case _:
                    raise ValueError("unknown pooling mode specified")

        return dCdx.reshape(self.input_vector_shape)
    
    def __str__(self):
        return f"pool {self._mode}: {self.input_shape} -> {self.output_shape} (0)"
