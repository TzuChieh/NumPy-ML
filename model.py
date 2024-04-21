"""
@brief Contains components for building a neural network.
All 1-D vectors  of `n` elements are assumed to have shape = `(n, 1)` (a column vector).
"""


import random
from abc import ABC, abstractmethod
from typing import Iterable
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
    

class ActivationFunction(ABC):
    @abstractmethod
    def eval(self, z):
        """
        @param The input vector.
        @return Vector of the evaluated function.
        """
        pass
        
    @abstractmethod
    def jacobian(self, z, **kwargs):
        """
        @param The input vector.
        @return A matrix of all the function's first-order derivatives with respect to `z`. For example,
        the 1st row would be (da(z1)/dz1, da(z1)/dz2, ..., da(z1)/dzn),
        the 2nd row would be (da(z2)/dz1, da(z2)/dz2, ..., da(z2)/dzn),
        and so on (all the way to da(zn)/dzn).
        """
        pass


class Layer(ABC):
    @property
    @abstractmethod
    def bias(self):
        """
        @return The bias vector.
        """
        pass

    @property
    @abstractmethod
    def weight(self):
        """
        @return The weight matrix.
        """
        pass

    @property
    @abstractmethod
    def activation(self) -> ActivationFunction:
        """
        @return The activation function.
        """
        pass

    @abstractmethod
    def weighted_input(self, input_a):
        """
        @param input_a The input activation vector.
        @return The weighted input vector (z).
        """
        pass

    @abstractmethod
    def update_parameters(self, bias, weight):
        """
        @param bias The new bias vector.
        @param weight The new weight matrix.
        """
        pass

    @abstractmethod
    def derived_parameters(self, input_a, delta):
        """
        @param input_a The input activation vector.
        @param delta The error vector.
        @return Gradient of biase and weight in the form `(del_b, del_w)`.
        """
        pass

    @abstractmethod
    def feedforward(self, input_a, **kwargs):
        """
        @param input_a The input activation vector.
        @param kwargs Allows to specify implementation defined extra arguments.
        @return Activation vector of the layer.
        """
        pass

    @abstractmethod
    def backpropagate(self, delta):
        """
        @param delta The error vector.
        @return The error vector for previous layer.
        """
        pass


class CostFunction(ABC):
    @abstractmethod
    def eval(self, a, y):
        """
        @return A vector of cost values.
        """
        pass

    @abstractmethod
    def derived_eval(self, a, y):
        """
        This method would have return a jacobian like `ActivationFunction.jacobian()`. However, currently all
        implementations are element-wise independent so we stick with returning a vector.
        @return A vector of derived cost values.
        """
        pass


class SigmoidActivation(ActivationFunction):
    def eval(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def jacobian(self, z, **kwargs):
        # Sigmoid is element-wise independent, so its jacobian is simply a diagonal matrix
        a = kwargs['a'] if 'a' in kwargs else self.eval(z)
        dadz = a * (1 - a)
        return np.diagflat(dadz)
    

class SoftmaxActivation(ActivationFunction):
    def eval(self, z):
        # Improves numerical stability (does not change the result--will cancel out in the division)
        z = z - np.max(z, axis=0, keepdims=True)
        e_z = np.exp(z)
        a = e_z / np.sum(e_z, axis=0, keepdims=True)
        return a
    
    def jacobian(self, z, **kwargs):
        # For a derivation that is clean and avoids looping & branching, see
        # https://mattpetersen.github.io/softmax-with-cross-entropy
        a = kwargs['a'] if 'a' in kwargs else self.eval(z)
        dadz = np.diagflat(a) - np.outer(a, a)
        return dadz


class QuadraticCost(CostFunction):
    def eval(self, a, y):
        """
        Basically computing the MSE between activations `a` and desired output `y`. The 0.5 multiplier is
        to make its derived form cleaner (for convenience).
        """
        C = np.float32(0.5) * np.linalg.norm(a - y) ** 2
        return C
    
    def derived_eval(self, a, y):
        dCda = a - y
        return dCda

    def delta(self, a, y, z, output_layer):
        dCda = a - y
        dadz = output_layer.activation.jacobian(z, a=a)
        return dadz @ dCda
    

class CrossEntropyCost(CostFunction):
    def eval(self, a, y):
        # Note that the `(1 - y) * log(1 - a)` term can be NaN/Inf, and `np.nan_to_num()` can help with that
        C = np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
        return C

    def derived_eval(self, a, y):
        # Calculates `(a - y) / (a * (1 - a))` without 0/0 division
        rcp_deno = np.nan_to_num(np.reciprocal(a * (1 - a)))
        dCda = (a - y) * rcp_deno
        return dCda


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size, activation_type: ActivationFunction=SigmoidActivation):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._activation = activation_type()
        self.init_scaled_gaussian_weights()

    @property
    def bias(self):
        return self._bias
    
    @property
    def weight(self):
        return self._weight
    
    @property
    def activation(self):
        return self._activation
    
    def weighted_input(self, input_a):
        x = input_a
        b = self._bias
        w = self._weight
        return w @ x + b
    
    def update_parameters(self, bias, weight):
        assert self._bias.dtype == 'float32'
        assert self._weight.dtype == 'float32'

        self._bias = bias
        self._weight = weight

    def derived_parameters(self, input_a, delta):
        del_b = np.copy(delta)
        del_w = delta @ input_a.T
        return (del_b, del_w)

    def feedforward(self, input_a, **kwargs):
        """
        @param kwargs 'z': weighted input (`input_a` will be ignored).
        """
        z = kwargs['z'] if 'z' in kwargs else self.weighted_input(input_a)
        return self.activation.eval(z)

    def backpropagate(self, delta):
        assert delta.dtype == 'float32'

        w_T = self._weight.T
        return w_T @ delta

    def init_gaussian_weights(self):
        """
        Included just for completeness. Please use at least `init_scaled_gaussian_weights()` for better performance.
        """
        rng = np.random.default_rng()
        y = self._output_size
        x = self._input_size
        self._bias = rng.standard_normal((y, 1), dtype=np.float32)
        self._weight = rng.standard_normal((y, x), dtype=np.float32)

    def init_scaled_gaussian_weights(self):
        rng = np.random.default_rng()
        y = self._output_size
        x = self._input_size
        self._bias = rng.standard_normal((y, 1), dtype=np.float32)
        self._weight = rng.standard_normal((y, x), dtype=np.float32) / np.sqrt(x, dtype=np.float32)


class Network:
    def __init__(self, hidden_layers: Iterable[Layer], cost_type=CrossEntropyCost):
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers) + 1
        self.cost = cost_type()
        self.init_velocities()

    def feedforward(self, x):
        """
        @param x The input vector.
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
        eta,
        momentum = 0.0,
        lambba = 0.0,
        test_data = None):
        """
        @param eta Learning rate.
        @param lambba The regularization parameter.
        """
        sgd_start_time = timer()

        eta = np.float32(eta)
        momentum = np.float32(momentum)
        lambba = np.float32(lambba)

        for ei in range(num_epochs):
            epoch_start_time = timer()

            random.shuffle(training_data)
            mini_batches = [
                training_data[bi : bi + mini_batch_size]
                for bi in range(0, len(training_data), mini_batch_size)]
            for mini_batch_data in mini_batches:
                self._update_mini_batch(mini_batch_data, eta, momentum, lambba, len(training_data))
            
            if test_data:
                performance_info = f"; performance: {self.performance_report(test_data)}"
            else:
                performance_info = f"; performance evaluation skipped"

            performance_info += f"; Δt: {timedelta(seconds=(timer() - epoch_start_time))}"

            print(f"epoch {ei + 1} / {num_epochs}{performance_info}")

        print(f"SGD Δt: {timedelta(seconds=(timer() - sgd_start_time))}")

    def performance_report(self, test_data):
        """
        @return A string that contains information of the network performance.
        """
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in test_data]
        num_right_answers = sum(1 for network_y, y in results if network_y == y)
        num_test_data = len(test_data)
        return f"{num_right_answers} / {num_test_data} ({num_right_answers / num_test_data})"

    def init_velocities(self):
        """
        Initialize gradient records.
        """
        self.v_biases = [np.zeros(b.shape, dtype=np.float32) for b in self.biases]
        self.v_weights = [np.zeros(w.shape, dtype=np.float32) for w in self.weights]

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

    def _update_mini_batch(self, mini_batch_data, eta, momentum, lambba, n):
        """
        @param n Number of training samples. To see why dividing by `n` is used for regularization,
        see https://datascience.stackexchange.com/questions/57271/why-do-we-divide-the-regularization-term-by-the-number-of-examples-in-regularize.
        """
        n = np.float32(n)
        mini_batch_n = np.float32(len(mini_batch_data))

        # Approximating the true `del_b` and `del_w` from m samples for each hidden layer
        del_bs = [np.zeros(layer.bias.shape, dtype=np.float32) for layer in self.hidden_layers]
        del_ws = [np.zeros(layer.weight.shape, dtype=np.float32) for layer in self.hidden_layers]
        for x, y in mini_batch_data:
            delta_del_bs, delta_del_ws = self._backpropagation(x, y)
            del_bs = [bi + dbi for bi, dbi in zip(del_bs, delta_del_bs)]
            del_ws = [wi + dwi for wi, dwi in zip(del_ws, delta_del_ws)]
        del_bs = [bi / mini_batch_n for bi in del_bs]
        del_ws = [wi / mini_batch_n for wi in del_ws]

        # Update momentum parameters
        self.v_biases = [
            vb * momentum - eta * bi for vb, bi in zip(self.v_biases, del_bs)]
        self.v_weights = [
            vw * momentum - eta * lambba / n * w - eta * wi for w, vw, wi in zip(self.weights, self.v_weights, del_ws)]

        # Update biases and weights
        new_biases = [b + vb for b, vb in zip(self.biases, self.v_biases)]
        new_weights = [w + vw for w, vw in zip(self.weights, self.v_weights)]
        for layer, new_b, new_w in zip(self.hidden_layers, new_biases, new_weights):
            layer.update_parameters(new_b, new_w)

    def _backpropagation(self, x, y):
        """
        @param x Training inputs.
        @param y Training outputs.
        """
        del_bs = [np.zeros(b.shape, dtype=np.float32) for b in self.biases]
        del_ws = [np.zeros(w.shape, dtype=np.float32) for w in self.weights]
        
        # Forward pass: store activations & weighted inputs (`zs`) layer by layer
        activations = [x]
        zs = []
        for layer in self.hidden_layers:
            z = layer.weighted_input(activations[-1])
            zs.append(z)
            activations.append(layer.feedforward(None, z=z))

        # Backward pass (initial delta term, must take cost function into account)
        dCda = self.cost.derived_eval(activations[-1], y)
        dadz = self.hidden_layers[-1].activation.jacobian(None, a=activations[-1])
        delta = dadz.T @ dCda

        # Backward pass (hidden layers)
        del_bs[-1], del_ws[-1] = self.hidden_layers[-1].derived_parameters(activations[-2], delta)
        for layer_idx in reversed(range(0, self.num_layers - 2)):
            layer = self.hidden_layers[layer_idx]
            next_layer = self.hidden_layers[layer_idx + 1]
            delta = next_layer.backpropagate(delta)
            dadz = layer.activation.jacobian(zs[layer_idx])
            delta = dadz.T @ delta
            del_bs[layer_idx], del_ws[layer_idx] = layer.derived_parameters(activations[layer_idx], delta)

        return (del_bs, del_ws)
