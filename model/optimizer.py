"""
@brief Contains components for building a neural network.
All 1-D vectors  of `n` elements are assumed to have shape = `(n, 1)` (a column vector).
"""


import common as com
import common.vector as vec
import common.progress_bar as progress_bar
from model.network import Network

import numpy as np

import random
import sys
from abc import ABC, abstractmethod
from timeit import default_timer as timer
from datetime import timedelta
from concurrent import futures


class Optimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        network: Network,
        num_epochs,
        training_data,
        print_progress=False):
        """
        Train the given network.
        """
        pass

    @abstractmethod
    def total_cost(self, network: Network, dataset):
        """
        Compute total cost of the network for a given dataset. Do not use this for training.
        @param dataset A list of pairs. Each pair contains input activation (x) and output activation (y).
        @return Total cost of the dataset with respect to the view of this optimizer.
        """
        pass

    @abstractmethod
    def get_info(self) -> str:
        """
        @return A string containing general information of this optimizer.
        """
        pass

    @property
    @abstractmethod
    def train_time(self) -> timedelta:
        """
        @return Time spent for the last call to `optimize()`.
        """
        pass
    
    @property
    @abstractmethod
    def total_train_time(self) -> timedelta:
        """
        @return Time spent for all calls to `optimize()`.
        """
        pass

    @property
    @abstractmethod
    def total_epochs(self) -> int:
        """
        @return Total number of epochs done.
        """
        pass


class StochasticGradientDescent(Optimizer):
    def __init__(
        self,
        mini_batch_size,
        gradient_clip_norm=sys.float_info.max,
        momentum=0.0,
        eta=1,
        lambba=0.0,
        num_workers=0):
        """
        @param gradient_clip_norm Clip threshold for backpropagation gradient based on norm.
        @param eta Learning rate.
        @param lambba The regularization parameter.
        """
        super().__init__()

        self._num_workers = num_workers
        # self._workers = None if num_workers <= 1 else futures.ThreadPoolExecutor(max_workers=num_workers)
        self._workers = None if num_workers <= 1 else futures.ProcessPoolExecutor(max_workers=num_workers)
        self._mini_batch_size = mini_batch_size
        self._gradient_clip_norm = com.REAL_TYPE(np.minimum(gradient_clip_norm, np.finfo(com.REAL_TYPE).max))
        self._momentum = com.REAL_TYPE(momentum)
        self._eta = com.REAL_TYPE(eta)
        self._lambba = com.REAL_TYPE(lambba)
        self._train_time = timedelta(seconds=0)
        self._total_train_time = timedelta(seconds=0)
        self._total_epochs = 0
        self.v_biases = None
        self.v_weights = None

    def __del__(self):
        if self._workers is not None:
            self._workers.shutdown()
        
    def optimize(
        self,
        network: Network,
        num_epochs,
        training_data,
        print_progress=False):
        """
        @param network The network to train.
        """
        start_time = timer()

        self._prepare_param_velocities(network)
        for _ in range(num_epochs):
            # Prepare mini batch data
            random.shuffle(training_data)
            mini_batches = [
                training_data[bi : bi + self._mini_batch_size]
                for bi in range(0, len(training_data), self._mini_batch_size)]
            
            if print_progress:
                self._print_progress(0)

            # Train with mini batches
            for bi, mini_batch_data in enumerate(mini_batches):
                self._mini_batch_param_update(network, mini_batch_data, len(training_data))

                if print_progress:
                    ms = timedelta(seconds=(timer() - start_time)).total_seconds() / (bi + 1) * 1000
                    self._print_progress((bi + 1) / len(mini_batches), suffix=f" {ms:.2f} ms/batch ")

            self._total_epochs += 1

        self._train_time = timedelta(seconds=(timer() - start_time))
        self._total_train_time += self._train_time

    def total_cost(self, network: Network, dataset):
        cost = 0.0
        rcp_dataset_n = 1.0 / len(dataset)
        for x, y in dataset:
            a = network.feedforward(vec.vector_2d(x), is_training=False)
            cost += network.cost.eval(a, vec.vector_2d(y)) * rcp_dataset_n

        # L2 regularization term
        sum_w2 = sum(np.linalg.norm(w) ** 2 for w in network.weights)
        cost += self._lambba * rcp_dataset_n * 0.5 * sum_w2

        return cost

    def get_info(self):
        info = f"optimizer: stochastic gradient descent\n"
        info += f"mini batch size: {self._mini_batch_size}\n"
        info += f"workers: {self._num_workers}\n"
        info += f"gradient clip: {self._gradient_clip_norm}\n"
        info += f"momentum: {self._momentum}\n"
        info += f"learning rate: {self._eta}\n"
        info += f"L2 regularization: {self._lambba}"
        return info
    
    @property
    def train_time(self):
        """
        @return Time spent for the last call to `train()`.
        """
        return self._train_time
    
    @property
    def total_train_time(self):
        """
        @return Time spent for all calls to `train()`.
        """
        return self._total_train_time
    
    @property
    def total_epochs(self):
        return self._total_epochs

    def _mini_batch_param_update(self, network: Network, mini_batch_data, n):
        """
        @param n Number of training samples. To see why dividing by `n` is used for regularization,
        see https://datascience.stackexchange.com/questions/57271/why-do-we-divide-the-regularization-term-by-the-number-of-examples-in-regularize.
        """
        del_bs, del_ws = _sgd_mini_batch_backpropagation(
            mini_batch_data, network, self._gradient_clip_norm, self._workers)

        # Update momentum parameters
        rcp_n = com.REAL_TYPE(1.0 / n)
        self.v_biases = [
            vb * self._momentum - self._eta * bi
            for vb, bi in zip(self.v_biases, del_bs)]
        self.v_weights = [
            vw * self._momentum - self._eta * self._lambba * rcp_n * w - self._eta * wi
            for w, vw, wi in zip(network.weights, self.v_weights, del_ws)]

        # Compute new biases and weights
        new_biases = [b + vb for b, vb in zip(network.biases, self.v_biases)]
        new_weights = [w + vw for w, vw in zip(network.weights, self.v_weights)]

        # Update biases and weights
        for layer, new_b, new_w in zip(network.hidden_layers, new_biases, new_weights):
            layer.update_params(new_b, new_w)

    def _prepare_param_velocities(self, network: Network):
        """
        Initialize gradient records if not already exist.
        """
        if self.v_biases is None:
            self.v_biases = [vec.zeros_from(b) for b in network.biases]

        if self.v_weights is None:
            self.v_weights = [vec.zeros_from(w) for w in network.weights]

    def _print_progress(self, fraction, suffix=""):
        progress_bar.put(fraction, num_progress_chars=40, prefix="MBSGD: ", suffix=suffix)


def _sgd_backpropagation(
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
    cache = {}
    
    # Forward pass: store activations & weighted inputs (`zs`) layer by layer
    activations = [vec.vector_2d(x)]
    zs = [None]
    for layer in network.hidden_layers:
        z = layer.weighted_input(activations[-1], cache)
        zs.append(z)
        activations.append(layer.feedforward(activations[-1], cache))

    # Backward pass (initial delta term, must take cost function into account)
    dCda = network.cost.derived_eval(activations[-1], vec.vector_2d(y))
    delta = network.hidden_layers[-1].activation.jacobian_mul(zs[-1], right=dCda, a=activations[-1])
    delta = _gradient_clip(delta, gradient_clip_norm)

    # Backward pass (hidden layers)
    del_bs[-1], del_ws[-1] = network.hidden_layers[-1].derived_params(activations[-2], delta, cache)
    for layer_idx in reversed(range(0, network.num_layers - 2)):
        layer = network.hidden_layers[layer_idx]
        next_layer = network.hidden_layers[layer_idx + 1]
        delta = next_layer.backpropagate(activations[layer_idx + 1], delta, cache)
        delta = layer.activation.jacobian_mul(zs[layer_idx + 1], right=delta, a=activations[layer_idx + 1])
        delta = _gradient_clip(delta, gradient_clip_norm)
        del_bs[layer_idx], del_ws[layer_idx] = layer.derived_params(activations[layer_idx], delta, cache)

    return (del_bs, del_ws)

def _sgd_mini_batch_backpropagation(
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
            delta_del_bs, delta_del_ws = _sgd_backpropagation(x, y, network, gradient_clip_norm)
            del_bs = [bi + dbi for bi, dbi in zip(del_bs, delta_del_bs)]
            del_ws = [wi + dwi for wi, dwi in zip(del_ws, delta_del_ws)]
    else:
        backpropagation_args = (
            [x for x, _ in mini_batch_data],
            [y for _, y in mini_batch_data],
            [network] * len(mini_batch_data),
            [gradient_clip_norm] * len(mini_batch_data))
        
        for result in workers.map(_sgd_backpropagation, *backpropagation_args):
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
