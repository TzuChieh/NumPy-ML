"""
@brief Contains components for building a neural network.
All 1-D vectors  of `n` elements are assumed to have shape = `(n, 1)` (a column vector).
"""


import common as com
import common.vector as vec
from model.cost import CostFunction, CrossEntropy
from model.layer import Layer

import numpy as np
import numpy.typing as np_typing

import random
import sys
from typing import Iterable
from timeit import default_timer as timer
from datetime import timedelta
from concurrent import futures


class Network:
    def __init__(self, hidden_layers: Iterable[Layer], cost: CostFunction=CrossEntropy()):
        self._hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers) + 1
        self.cost = cost
        self.init_velocities()

        layer_info = ""
        total_params = 0
        for layer in self.hidden_layers:
            layer_info += f"{str(layer)}\n"
            total_params += layer.num_params
        print(f"Network layers ({total_params} parameters):\n{layer_info}")

        if not _has_valid_connections(self.hidden_layers):
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
        report_eval_cost=False,
        num_workers=0):
        """
        @param gradient_clip_norm Clip threshold for backpropagation gradient based on norm.
        @param eta Learning rate.
        @param lambba The regularization parameter.
        """
        workers = None if num_workers <= 1 else futures.ProcessPoolExecutor(max_workers=num_workers)
        gradient_clip_norm = com.REAL_TYPE(np.minimum(gradient_clip_norm, np.finfo(com.REAL_TYPE).max))
        momentum = com.REAL_TYPE(momentum)
        eta = com.REAL_TYPE(eta)
        lambba = com.REAL_TYPE(lambba)

        print(f"optimizer: stochastic gradient descent")
        print(f"mini batch size: {mini_batch_size}, workers: {num_workers}")
        print(f"gradient clip: {gradient_clip_norm}, momentum: {momentum}")
        print(f"learning rate: {eta}, L2 regularization: {lambba}")

        sgd_start_time = timer()

        for ei in range(num_epochs):
            epoch_start_time = timer()

            random.shuffle(training_data)
            mini_batches = [
                training_data[bi : bi + mini_batch_size]
                for bi in range(0, len(training_data), mini_batch_size)]
            for mini_batch_data in mini_batches:
                self._update_mini_batch(
                    mini_batch_data,
                    eta,
                    gradient_clip_norm,
                    momentum,
                    lambba,
                    len(training_data),
                    workers)
            
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

        if workers is not None:
            workers.shutdown()

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
    
    @property
    def hidden_layers(self):
        """
        @return List of hidden layers in this network.
        """
        return self._hidden_layers

    def _update_mini_batch(
        self,
        mini_batch_data,
        eta,
        gradient_clip_norm,
        momentum,
        lambba,
        n,
        workers: futures.Executor):
        """
        @param n Number of training samples. To see why dividing by `n` is used for regularization,
        see https://datascience.stackexchange.com/questions/57271/why-do-we-divide-the-regularization-term-by-the-number-of-examples-in-regularize.
        """
        del_bs, del_ws = _mini_batch_backpropagation(mini_batch_data, self, gradient_clip_norm, workers)

        # Update momentum parameters
        rcp_n = com.REAL_TYPE(1.0 / n)
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


def _backpropagation(
    x,
    y,
    network: Network,
    gradient_clip_norm):
    """
    @param x Training inputs.
    @param y Training outputs.
    """
    del_bs = [vec.zeros_from(b) for b in network.biases]
    del_ws = [vec.zeros_from(w) for w in network.weights]
    
    # Forward pass: store activations & weighted inputs (`zs`) layer by layer
    activations = [vec.vector_2d(x)]
    zs = []
    for layer in network.hidden_layers:
        z = layer.weighted_input(activations[-1])
        zs.append(z)
        activations.append(layer.feedforward(activations[-1], z=z))

    # Backward pass (initial delta term, must take cost function into account)
    dCda = network.cost.derived_eval(activations[-1], vec.vector_2d(y))
    dadz = network.hidden_layers[-1].activation.jacobian(zs[-1], a=activations[-1])
    dadz_T = vec.transpose_2d(dadz)
    delta = dadz_T @ dCda
    delta = _gradient_clip(delta, gradient_clip_norm)

    # Backward pass (hidden layers)
    del_bs[-1], del_ws[-1] = network.hidden_layers[-1].derived_params(activations[-2], delta)
    for layer_idx in reversed(range(0, network.num_layers - 2)):
        layer = network.hidden_layers[layer_idx]
        next_layer = network.hidden_layers[layer_idx + 1]
        delta = next_layer.backpropagate(activations[layer_idx + 1], delta)
        dadz = layer.activation.jacobian(zs[layer_idx])
        dadz_T = vec.transpose_2d(dadz)
        delta = dadz_T @ delta
        delta = _gradient_clip(delta, gradient_clip_norm)
        del_bs[layer_idx], del_ws[layer_idx] = layer.derived_params(activations[layer_idx], delta)

    return (del_bs, del_ws)

def _mini_batch_backpropagation(
    mini_batch_data,
    network: Network,
    gradient_clip_norm,
    workers: futures.Executor):
    """
    Performs backpropagation for a collection of training data.
    @param mini_batch_data List of training data. Each element is a tuple of `(x, y)`.
    @param workers If not `None`, use the specified workers to perform the calculation.
    """
    rcp_mini_batch_n = com.REAL_TYPE(1.0 / len(mini_batch_data))

    del_bs = [vec.zeros_from(layer.bias) for layer in network.hidden_layers]
    del_ws = [vec.zeros_from(layer.weight) for layer in network.hidden_layers]

    # Approximating the true `del_b` and `del_w` from m samples for each hidden layer

    # Summation of gradients
    if workers is None:
        for x, y in mini_batch_data:
            delta_del_bs, delta_del_ws = _backpropagation(x, y, network, gradient_clip_norm)
            del_bs = [bi + dbi for bi, dbi in zip(del_bs, delta_del_bs)]
            del_ws = [wi + dwi for wi, dwi in zip(del_ws, delta_del_ws)]
    else:
        backpropagation_args = (
            [x for x, _ in mini_batch_data],
            [y for _, y in mini_batch_data],
            [network] * len(mini_batch_data),
            [gradient_clip_norm] * len(mini_batch_data))
        
        for result in workers.map(_backpropagation, *backpropagation_args):
            delta_del_bs, delta_del_ws = result
            del_bs = [bi + dbi for bi, dbi in zip(del_bs, delta_del_bs)]
            del_ws = [wi + dwi for wi, dwi in zip(del_ws, delta_del_ws)]

    # Averaging over m samples
    del_bs = [bi * rcp_mini_batch_n for bi in del_bs]
    del_ws = [wi * rcp_mini_batch_n for wi in del_ws]

    return (del_bs, del_ws)

def _gradient_clip(delta, gradient_clip_norm):
    # Potentially apply gradient clipping
    if gradient_clip_norm < np.finfo(com.REAL_TYPE).max:
        delta_norm = np.linalg.norm(delta)
        if delta_norm >= gradient_clip_norm:
            delta = delta / delta_norm * gradient_clip_norm
    return delta

def _has_valid_connections(layers: Iterable[Layer]):
    """
    @return Whether the specified layers properly connect with each other.
    """
    if len(layers) <= 1:
        return True

    for curr_layer, prev_layer in zip(layers[1:], layers[:-1]):
        if not np.array_equal(curr_layer.input_shape, prev_layer.output_shape):
            return False
    return True
