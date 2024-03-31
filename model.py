import random

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def derived_sigmoid(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

class Network:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        rng = np.random.default_rng()
        self.biases = [rng.standard_normal((y, 1)) for y in layer_sizes[1:]]
        self.weights = [rng.standard_normal((y, x)) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

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
                num_right_answers = self.evaluate_performance(test_data)
                num_test_data = len(test_data)
                performance_info = (
                    f"performance: {num_right_answers} / {num_test_data} ({num_right_answers / num_test_data})")
            else:
                performance_info = "performance evaluation skipped"

            print(f"epoch {ei} / {num_epochs}; {performance_info}")

    def evaluate_performance(self, test_data):
        """
        @return Number of correct outputs.
        """
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in test_data]
        return sum(1 for network_y, y in results if network_y == y)


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
        delta = self._derived_cost(activations[-1], y) * derived_sigmoid(zs[-1])
        del_b[-1] = delta
        del_w[-1] = np.matmul(delta, activations[-2].transpose())
        for layer_idx in reversed(range(0, self.num_layers - 2)):
            z = zs[layer_idx]
            derived_s = derived_sigmoid(z)
            delta = np.matmul(self.weights[layer_idx + 1].transpose(), delta) * derived_s
            del_b[layer_idx] = delta
            del_w[layer_idx] = np.matmul(delta, activations[layer_idx].transpose())

        return del_b, del_w
        
    def _derived_cost(self, output_activations, y):
        """
        @return @f$ \\partial C_x / \\partial a_output @f$
        """
        return output_activations - y
