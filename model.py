"""
@brief Contains components for building a neural network.
All 1-D vectors  of `n` elements are assumed to have shape = `(n, 1)` (a column vector).
"""


import vector as vec

import numpy as np
import numpy.typing as np_type

import random
import warnings
import sys
from abc import ABC, abstractmethod
from typing import Iterable
from timeit import default_timer as timer
from datetime import timedelta
    

# Type of the numbers used for calculation. Note that `real_type` can be used for both contructing scalars or for
# specifying `dtype` arguments. `real_dtype` should be used for comparing `dtype`s
real_type = np.float32
real_dtype = np.dtype(real_type)


# An approximative value of machine epsilon, which is useful in avoiding some numerical issues such as division
# by zero. Keras also uses this value, see https://github.com/tensorflow/tensorflow/blob/066e226b3ed6db054cdb5ed0ff2453b8c1ffb3f6/tensorflow/python/keras/backend_config.py#L24
epsilon = real_type(1e-7)


class ActivationFunction(ABC):
    @abstractmethod
    def eval(self, z):
        """
        @param z The input vector.
        @return Vector of the evaluated function.
        """
        pass
        
    @abstractmethod
    def jacobian(self, z, **kwargs):
        """
        @param z The input vector.
        @param kwargs Implementation defined extra arguments (e.g., to facilitate the calculation).
        @return A matrix of all the function's first-order derivatives with respect to `z`. For example,
        the 1st row would be (da(z1)/dz1, da(z1)/dz2, ..., da(z1)/dzn),
        the 2nd row would be (da(z2)/dz1, da(z2)/dz2, ..., da(z2)/dzn),
        and so on (all the way to da(zn)/dzn).
        """
        pass


class Layer(ABC):
    """
    Abstraction for a single layer in a neural network. Parameters should be of the defined shape (e.g., a vector)
    in the lowest dimensions, and be broadcastable to higher dimensions.
    """
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def bias(self) -> np_type.NDArray:
        """
        @return Bias parameters. `None` if there is no bias term.
        """
        pass

    @property
    @abstractmethod
    def weight(self) -> np_type.NDArray:
        """
        @return Weight parameters. `None` if there is no weight term.
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
    def weighted_input(self, x: np_type.NDArray):
        """
        @param x The input activation vector.
        @return The weighted input vector (z).
        """
        pass

    @abstractmethod
    def update_params(self, bias: np_type.NDArray, weight: np_type.NDArray):
        """
        @param bias The new bias parameters.
        @param weight The new weight parameters.
        """
        pass

    @abstractmethod
    def derived_params(self, x: np_type.NDArray, delta: np_type.NDArray):
        """
        @param x The input activation vector.
        @param delta The error vector (dCdz).
        @return Gradient of `bias` and `weight` in the form `(del_b, del_w)`.
        """
        pass

    @abstractmethod
    def feedforward(self, x: np_type.NDArray, **kwargs):
        """
        @param x The input activation vector.
        @param kwargs Implementation defined extra arguments (e.g., to facilitate the calculation).
        @return Activation vector of the layer.
        """
        pass

    @abstractmethod
    def backpropagate(self, x, delta: np_type.NDArray):
        """
        @param delta The error vector (dCdz).
        @return The error vector (dCdx).
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

    def init_normal_params(self):
        """
        Randomly set initial parameters. The random values forms a standard normal distribution.
        """
        rng = np.random.default_rng()
        b = rng.standard_normal(self.bias.shape, dtype=real_type)
        w = rng.standard_normal(self.weight.shape, dtype=real_type)
        self.update_params(b, w)

    def init_scaled_normal_params(self):
        """
        Scaled (weights) version of `init_normal_params()`. In theory works better for sigmoid and tanh neurons.
        """
        rng = np.random.default_rng()
        nx = self.input_vector_shape[-2]
        b = rng.standard_normal(self.bias.shape, dtype=real_type)
        w = rng.standard_normal(self.weight.shape, dtype=real_type) / np.sqrt(nx, dtype=real_type)
        self.update_params(b, w)


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


class Sigmoid(ActivationFunction):
    def eval(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def jacobian(self, z, **kwargs):
        # Sigmoid is element-wise independent, so its jacobian is simply a diagonal matrix
        a = kwargs['a'] if 'a' in kwargs else self.eval(z)
        dadz = a * (1 - a)
        return np.diagflat(dadz)
    

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
        dadz = np.diagflat(a) - np.outer(a, a)
        return dadz


class Tanh(ActivationFunction):
    def eval(self, z):
        return np.tanh(z)

    def jacobian(self, z, **kwargs):
        a = kwargs['a'] if 'a' in kwargs else self.eval(z)
        dadz = 1 - np.square(a)
        return np.diagflat(dadz)


class ReLU(ActivationFunction):
    def eval(self, z):
        return z * (z > 0)
    
    def jacobian(self, z, **kwargs):
        # Derivatives at 0 is implemented as 0. See "Numerical influence of ReLU'(0) on backpropagation",
        # https://hal.science/hal-03265059/file/Impact_of_ReLU_prime.pdf
        dadz = (z > 0).astype(real_type)
        return np.diagflat(dadz) 


class Quadratic(CostFunction):
    def eval(self, a, y):
        """
        Basically computing the MSE between activations `a` and desired output `y`. The 0.5 multiplier is
        to make its derived form cleaner (for convenience).
        """
        C = real_type(0.5) * np.linalg.norm(a - y) ** 2
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
        return np.clip(a, epsilon, 1 - epsilon)


class FullyConnected(Layer):
    """
    A fully connected network layer.
    """
    def __init__(
        self, 
        input_shape: Iterable[int],
        output_shape: Iterable[int], 
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
        self._bias = np.zeros((nc, ny, 1), dtype=real_type)
        self._weight = np.zeros((nc, ny, nx), dtype=real_type)

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

    def weighted_input(self, x):
        b = self._bias
        w = self._weight
        z = w @ x + b
        return z
    
    def update_params(self, bias, weight):
        assert self._bias.dtype == real_dtype, f"{self._bias.dtype}"
        assert self._weight.dtype == real_dtype, f"{self._weight.dtype}"

        self._bias = bias
        self._weight = weight

    def derived_params(self, x, delta):
        del_b = np.copy(delta)
        x_T = vec.transpose_2d(x)
        del_w = delta @ x_T
        return (del_b, del_w)

    def feedforward(self, x, **kwargs):
        """
        @param kwargs 'z': `weighted_input` of this layer (`x` will be ignored).
        """
        z = kwargs['z'] if 'z' in kwargs else self.weighted_input(x)
        return self.activation.eval(z)

    def backpropagate(self, x, delta):
        assert delta.dtype == real_dtype, f"{delta.dtype}"

        w_T = vec.transpose_2d(self._weight)
        dCdz = w_T @ delta
        return dCdz
    
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
        input_shape: Iterable[int],
        kernel_shape: Iterable[int],
        output_features: int,
        stride_shape: Iterable[int]=(1,), 
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
        self._bias = np.zeros((output_features, 1, 1, 1), dtype=real_dtype)
        self._weight = np.zeros((output_features, *kernel_shape), dtype=real_type)

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
    
    def weighted_input(self, x):
        x = x.reshape(self.input_shape)

        z = np.zeros(self.output_shape, dtype=x.dtype)
        for output_channel in range(self.output_shape[-3]):
            b = self._bias[output_channel]
            k = self._weight[output_channel]
            z[output_channel] = vec.correlate(x, k, self._stride_shape) + b

        return z.reshape(self.output_vector_shape)
    
    def update_params(self, bias, weight):
        self._bias = bias.reshape(self._bias.shape)
        self._weight = weight.reshape(self._weight.shape)

    def derived_params(self, x, delta):
        x = x.reshape(self.input_shape)
        delta = delta.reshape(self.output_shape)

        sum_axes = tuple(di for di in range(-len(self._kernel_shape), 0))
        del_b = delta.sum(axis=sum_axes, keepdims=True, dtype=real_type).reshape(self.bias.shape)

        # Backpropagation is equivalent to a stride-1 correlation of input with a dilated gradient
        dilated_delta = vec.dilate(delta, self._stride_shape)
        del_w = vec.zeros_from(self.weight)
        for output_channel in range(self.output_shape[-3]):
            d = dilated_delta[np.newaxis, output_channel, ...]
            del_w[output_channel] = vec.correlate(x, d)

        return (del_b, del_w)

    def feedforward(self, x, **kwargs):
        """
        @param kwargs 'z': `weighted_input` of this layer (`x` will be ignored).
        """
        z = kwargs['z'] if 'z' in kwargs else self.weighted_input(x)
        return self.activation.eval(z)

    def backpropagate(self, x, delta):
        assert delta.dtype == real_dtype, f"{delta.dtype}"

        delta = delta.reshape(self.output_shape)

        # Backpropagation is equivalent to a stride-1 full correlation of a dilated (and padded) gradient with
        # a reversed kernel
        flip_axes = tuple(di for di in range(-len(self._kernel_shape), 0))
        reversed_k = np.flip(self._weight, axis=flip_axes)
        pad_shape = np.subtract(self._kernel_shape, 1)
        dilated_delta = vec.dilate(delta, self._stride_shape, pad_shape=pad_shape)
        dCdz = np.zeros(self.input_shape, dtype=delta.dtype)
        for output_channel in range(self.output_shape[-3]):
            k = reversed_k[output_channel]
            d = dilated_delta[np.newaxis, output_channel, ...]
            dCdz += vec.correlate(d, k)

        return dCdz.reshape(self.input_vector_shape)
    
    def __str__(self):
        kernel_info = "x".join(str(ks) for ks in self._kernel_shape)
        return f"{kernel_info} convolution: {self.input_shape} -> {self.output_shape} ({self.num_params})"


class Pool(Layer):
    """
    A max pooling layer.
    """
    def __init__(
        self, 
        input_shape: Iterable[int],
        kernel_shape: Iterable[int],
        mode: vec.PoolingMode=vec.PoolingMode.MAX,
        stride_shape: Iterable[int]=None):
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
        self._rcp_num_kernel_elements = np.reciprocal(self._kernel_shape.prod(), dtype=real_type)
        self._mode = mode
        self._stride_shape = np.array(stride_shape)

    @property
    def bias(self):
        return None
    
    @property
    def weight(self):
        return None
    
    @property
    def activation(self):
        return None
    
    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape
    
    def weighted_input(self, x):
        x = x.reshape(self.input_shape)
        z = vec.pool(x, self._kernel_shape, self._stride_shape, self._mode)

        assert np.array_equal(z.shape, self.output_shape), f"shapes: {z.shape}, {self.output_shape}"
        return z
    
    def update_params(self, bias, weight):
        pass

    def derived_params(self, x, delta):
        return (None, None)

    def feedforward(self, x, **kwargs):
        """
        @param kwargs 'z': `weighted_input` of this layer (`x` will be ignored).
        """
        z = kwargs['z'] if 'z' in kwargs else self.weighted_input(x)
        return z

    def backpropagate(self, x, delta):
        assert delta.dtype == real_dtype, f"{delta.dtype}"

        x = delta.reshape(self.input_shape)
        delta = delta.reshape(self.output_shape)
        x_pool_view = vec.sliding_window_view(x, self._kernel_shape, self._stride_shape)

        dCdx = np.zeros(self.input_shape, dtype=delta.dtype)
        dCdx_pool_view = vec.sliding_window_view(dCdx, self._kernel_shape, self._stride_shape, is_writeable=True)
        for pool_idx in np.ndindex(self.output_shape):
            pool_idx = np.array(pool_idx)
            x_pool = x_pool_view[*pool_idx]
            dCdx_pool = dCdx_pool_view[*pool_idx]

            # Lots of indexing here, make sure we are getting expected shapes
            assert np.array_equal(x_pool.shape, self._kernel_shape), f"shapes: {x_pool.shape}, {self._kernel_shape}"
            assert np.array_equal(x_pool.shape, dCdx_pool.shape), f"shapes: {x_pool.shape}, {dCdx_pool.shape}"
            
            match self._mode:
                # Gradient only propagate to the max element (think of an imaginary weight of 1, non-max element
                # has 0 weight)
                case vec.PoolingMode.MAX:
                    dCdx_pool.flat[x_pool.argmax()] = delta[pool_idx]
                # Similar to the case of max pooling, average pooling is equivalent to an imaginary weight of
                # the reciprocal of number of pool elements
                case vec.PoolingMode.AVERAGE:
                    dCdx_pool += delta[pool_idx] * self._rcp_num_kernel_elements
                case _:
                    raise ValueError("unknown pooling mode specified")

        return dCdx.reshape(self.input_vector_shape)
    
    def __str__(self):
        return f"pool {self._mode}: {self.input_shape} -> {self.output_shape} (0)"


class Network:
    def __init__(self, hidden_layers: Iterable[Layer], cost: CostFunction=CrossEntropy()):
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers) + 1
        self.cost = cost
        self.init_velocities()

        layer_info = ""
        total_params = 0
        for layer in self.hidden_layers:
            layer_info += f"{str(layer)}\n"
            total_params += layer.num_params
        print(f"Network layers ({total_params} parameters):\n{layer_info}")

        if not self._has_valid_connections():
            raise ValueError("network layers does not have valid connections")

    def feedforward(self, x):
        """
        @param x The input vector.
        @return The output vector.
        """
        a = x
        for layer in self.hidden_layers:
            a = layer.feedforward(a)
        return a
    
    def stochastic_gradient_descent(
        self,
        training_data, 
        num_epochs,
        mini_batch_size,
        gradient_clip_norm=sys.float_info.max,
        momentum=0.0,
        eta=1,
        lambba=0.0,
        eval_data=None,
        report_training_performance=False,
        report_eval_performance=False,
        report_training_cost=False,
        report_eval_cost=False):
        """
        @param gradient_clip_norm Clip threshold for backpropagation gradient based on norm.
        @param eta Learning rate.
        @param lambba The regularization parameter.
        """
        gradient_clip_norm = real_type(np.minimum(gradient_clip_norm, np.finfo(real_type).max))
        momentum = real_type(momentum)
        eta = real_type(eta)
        lambba = real_type(lambba)

        print(f"optimizer: stochastic gradient descent")
        print(f"mini batch size: {mini_batch_size}, gradient clip: {gradient_clip_norm}, momentum: {momentum}")
        print(f"learning rate: {eta}, L2 regularization: {lambba}")

        sgd_start_time = timer()

        for ei in range(num_epochs):
            epoch_start_time = timer()

            random.shuffle(training_data)
            mini_batches = [
                training_data[bi : bi + mini_batch_size]
                for bi in range(0, len(training_data), mini_batch_size)]
            for mini_batch_data in mini_batches:
                self._update_mini_batch(mini_batch_data, eta, gradient_clip_norm, momentum, lambba, len(training_data))
            
            # Collects a brief report for this epoch
            report = ""

            if report_training_performance:
                num, frac = self.performance(training_data)
                report += f"; training perf: {num} / {len(training_data)} ({frac})"

            if report_eval_performance:
                assert eval_data is not None
                num, frac = self.performance(eval_data)
                report += f"; eval perf: {num} / {len(eval_data)} ({frac})"

            if report_training_cost:
                report += f"; training cost: {self.total_cost(training_data, lambba)}"

            if report_eval_cost:
                assert eval_data is not None
                report += f"; eval cost: {self.total_cost(eval_data, lambba)}"

            report += f"; Δt: {timedelta(seconds=(timer() - epoch_start_time))}"

            print(f"epoch {ei + 1} / {num_epochs}{report}")

        print(f"total Δt: {timedelta(seconds=(timer() - sgd_start_time))}")

    def performance(self, dataset):
        """
        @param dataset A list of pairs. Each pair contains input activation (x) and output activation (y).
        @return A tuple that contains (in order): number of correct outputs, fraction of correct outputs.
        """
        results = [(np.argmax(self.feedforward(vec.vector_2d(x))), np.argmax(y)) for x, y in dataset]
        num_right_answers = sum(1 for network_y, y in results if network_y == y)
        num_data = len(dataset)
        return (num_right_answers, num_right_answers / num_data)
    
    def total_cost(self, dataset, lambba):
        """
        @param dataset A list of pairs. Each pair contains input activation (x) and output activation (y).
        @param lambba The regularization parameter.
        @return Total cost of the dataset.
        """
        cost = 0.0
        rcp_dataset_n = 1.0 / len(dataset)
        for x, y in dataset:
            a = self.feedforward(vec.vector_2d(x))
            cost += self.cost.eval(a, vec.vector_2d(y)) * rcp_dataset_n

        # L2 regularization term
        sum_w2 = sum(np.linalg.norm(w) ** 2 for w in self.weights)
        cost += lambba * rcp_dataset_n * 0.5 * sum_w2

        return cost

    def init_velocities(self):
        """
        Initialize gradient records.
        """
        self.v_biases = [vec.zeros_from(b) for b in self.biases]
        self.v_weights = [vec.zeros_from(w) for w in self.weights]

    @property
    def biases(self):
        """
        @return List of biases of the hidden layers.
        """
        return [layer.bias for layer in self.hidden_layers]
    
    @property
    def weights(self):
        """
        @return List of weights of the hidden layers.
        """
        return [layer.weight for layer in self.hidden_layers]

    def _update_mini_batch(self, mini_batch_data, eta, gradient_clip_norm, momentum, lambba, n):
        """
        @param n Number of training samples. To see why dividing by `n` is used for regularization,
        see https://datascience.stackexchange.com/questions/57271/why-do-we-divide-the-regularization-term-by-the-number-of-examples-in-regularize.
        """
        rcp_n = real_type(1.0 / n)
        rcp_mini_batch_n = real_type(1.0 / len(mini_batch_data))

        # Approximating the true `del_b` and `del_w` from m samples for each hidden layer
        del_bs = [vec.zeros_from(layer.bias) for layer in self.hidden_layers]
        del_ws = [vec.zeros_from(layer.weight) for layer in self.hidden_layers]
        for x, y in mini_batch_data:
            delta_del_bs, delta_del_ws = self._backpropagation(x, y, gradient_clip_norm)
            del_bs = [bi + dbi for bi, dbi in zip(del_bs, delta_del_bs)]
            del_ws = [wi + dwi for wi, dwi in zip(del_ws, delta_del_ws)]
        del_bs = [bi * rcp_mini_batch_n for bi in del_bs]
        del_ws = [wi * rcp_mini_batch_n for wi in del_ws]

        # Update momentum parameters
        self.v_biases = [
            vb * momentum - eta * bi for vb, bi in zip(self.v_biases, del_bs)]
        self.v_weights = [
            vw * momentum - eta * lambba * rcp_n * w - eta * wi for w, vw, wi in zip(self.weights, self.v_weights, del_ws)]

        # Compute new biases and weights
        new_biases = [b + vb for b, vb in zip(self.biases, self.v_biases)]
        new_weights = [w + vw for w, vw in zip(self.weights, self.v_weights)]

        # Update biases and weights
        for layer, new_b, new_w in zip(self.hidden_layers, new_biases, new_weights):
            layer.update_params(new_b, new_w)

    def _backpropagation(self, x, y, gradient_clip_norm):
        """
        @param x Training inputs.
        @param y Training outputs.
        """
        del_bs = [vec.zeros_from(b) for b in self.biases]
        del_ws = [vec.zeros_from(w) for w in self.weights]
        
        # Forward pass: store activations & weighted inputs (`zs`) layer by layer
        activations = [vec.vector_2d(x)]
        zs = []
        for layer in self.hidden_layers:
            z = layer.weighted_input(activations[-1])
            zs.append(z)
            activations.append(layer.feedforward(activations[-1], z=z))

        # Backward pass (initial delta term, must take cost function into account)
        dCda = self.cost.derived_eval(activations[-1], vec.vector_2d(y))
        dadz = self.hidden_layers[-1].activation.jacobian(zs[-1], a=activations[-1])
        dadz_T = vec.transpose_2d(dadz)
        delta = dadz_T @ dCda
        delta = self._gradient_clip(delta, gradient_clip_norm)

        # Backward pass (hidden layers)
        del_bs[-1], del_ws[-1] = self.hidden_layers[-1].derived_params(activations[-2], delta)
        for layer_idx in reversed(range(0, self.num_layers - 2)):
            layer = self.hidden_layers[layer_idx]
            next_layer = self.hidden_layers[layer_idx + 1]
            delta = next_layer.backpropagate(activations[layer_idx + 1], delta)
            dadz = layer.activation.jacobian(zs[layer_idx])
            dadz_T = vec.transpose_2d(dadz)
            delta = dadz_T @ delta
            delta = self._gradient_clip(delta, gradient_clip_norm)
            del_bs[layer_idx], del_ws[layer_idx] = layer.derived_params(activations[layer_idx], delta)

        return (del_bs, del_ws)

    def _gradient_clip(self, delta, gradient_clip_norm):
        # Potentially apply gradient clipping
        if gradient_clip_norm < np.finfo(real_type).max:
            delta_norm = np.linalg.norm(delta)
            if delta_norm >= gradient_clip_norm:
                delta = delta / delta_norm * gradient_clip_norm
                print(f"clipped {delta}")
        return delta
    
    def _has_valid_connections(self):
        if len(self.hidden_layers) <= 1:
            return True

        for layer, prev_layer in zip(self.hidden_layers[1:], self.hidden_layers[:-1]):
            if not np.array_equal(layer.input_shape, prev_layer.output_shape):
                return False
        return True
