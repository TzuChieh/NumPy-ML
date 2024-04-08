import random
from abc import ABC, abstractmethod

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def derived_sigmoid(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

class CostFunction(ABC):
    @abstractmethod
    def func(self, a, y):
        pass

    @abstractmethod
    def delta(self, a, y, z):
        pass

class QuadraticCost(CostFunction):
    def func(self, a, y):
        """
        Basically computing the MSE between activations `a` and desired output `y`. The 0.5 multiplier is
        to make its derived form cleaner (for convenience).
        """
        return 0.5 * np.linalg.norm(a - y) ** 2
    
    def delta(self, a, y, z):
        """
        Note that the `(a - y)` term is @f$ \\partial C_x / \\partial a_output @f$.
        """
        return (a - y) * derived_sigmoid(z)
    
# class CrossEntropyCost(CostFunction):
#     def func(self, a, y):


class Network:
    def __init__(self, layer_sizes, cost_type=QuadraticCost):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.cost = cost_type()
        self.init_scaled_gaussian_weights()

    def feedforward(self, a):
        """
        @param a The input matrix. Note that a 1-D vector of `n` elements should have shape = `(n, 1)`
        (a column vector).
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.matmul(w, a) + b)
        return a
    
    def stochastic_gradient_descent(
        self,
        training_data, 
        num_epochs,
        mini_batch_size,
        eta,
        test_data = None):
        """
        @param eta Learning rate.
        """
        for ei in range(num_epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[bi : bi + mini_batch_size]
                for bi in range(0, len(training_data), mini_batch_size)]
            for mini_batch_data in mini_batches:
                self._update_mini_batch(mini_batch_data, eta)
            
            if test_data:
                performance_info = f"performance: {self.performance_report(test_data)}"
            else:
                performance_info = "performance evaluation skipped"

            print(f"epoch {ei + 1} / {num_epochs}; {performance_info}")

    def performance_report(self, test_data):
        """
        @return A string that contains information of the network performance.
        """
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in test_data]
        num_right_answers = sum(1 for network_y, y in results if network_y == y)
        num_test_data = len(test_data)
        return f"{num_right_answers} / {num_test_data} ({num_right_answers / num_test_data})"

    def init_gaussian_weights(self):
        """
        Included just for completeness. Please use at least `init_scaled_gaussian_weights()` for better performance.
        """
        rng = np.random.default_rng()
        sizes = self.layer_sizes
        self.biases = [rng.standard_normal((y, 1)) for y in sizes[1:]]
        self.weights = [rng.standard_normal((y, x)) for x, y in zip(sizes[:-1], sizes[1:])]

    def init_scaled_gaussian_weights(self):
        rng = np.random.default_rng()
        sizes = self.layer_sizes
        self.biases = [rng.standard_normal((y, 1)) for y in sizes[1:]]
        self.weights = [rng.standard_normal((y, x)) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

    def _update_mini_batch(self, mini_batch_data, eta):
        # Approximating the true `del_b` and `del_w` from m samples
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch_data:
            delta_del_b, delta_del_w = self._backpropagation(x, y)
            del_b = [bi + dbi for bi, dbi in zip(del_b, delta_del_b)]
            del_w = [wi + dwi for wi, dwi in zip(del_w, delta_del_w)]
        del_b = [bi / len(mini_batch_data) for bi in del_b]
        del_w = [wi / len(mini_batch_data) for wi in del_w]

        # Update biases and weights
        self.biases = [b - eta * bi for b, bi in zip(self.biases, del_b)]
        self.weights = [w - eta * wi for w, wi in zip(self.weights, del_w)]

    def _backpropagation(self, x, y):
        """
        @param x Training inputs.
        @param y Training outputs.
        """
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]

        # Forward pass: store activations & weighted inputs (`zs`) layer by layer
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.matmul(w, activations[-1]) + b
            zs.append(z)
            activations.append(sigmoid(z))

        # Backward pass
        delta = self.cost.delta(activations[-1], y, zs[-1])
        del_b[-1] = delta
        del_w[-1] = np.matmul(delta, activations[-2].transpose())
        for layer_idx in reversed(range(0, self.num_layers - 2)):
            z = zs[layer_idx]
            derived_s = derived_sigmoid(z)
            delta = np.matmul(self.weights[layer_idx + 1].transpose(), delta) * derived_s
            del_b[layer_idx] = delta
            del_w[layer_idx] = np.matmul(delta, activations[layer_idx].transpose())

        return del_b, del_w
