"""
@brief Contains components for building a neural network.
All 1-D vectors  of `n` elements are assumed to have shape = `(n, 1)` (a column vector).
"""


import common as com
import common.vector as vec
import common.progress_bar as progress_bar
from dataset import Dataset
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
        training_set: Dataset,
        print_progress=False):
        """
        Train the given network.
        """
        pass

    @abstractmethod
    def total_cost(self, dataset: Dataset, network: Network):
        """
        Compute total cost of the network for a given dataset. Do not use this for training.
        @param dataset A list of pairs. Each pair contains input activation (x) and output activation (y).
        @return Total cost of the dataset with respect to the view of this optimizer.
        """
        pass

    @abstractmethod
    def performance(self, dataset: Dataset, network: Network):
        """
        Evaluate the effectiveness of the network.
        @param dataset A list of pairs. Each pair contains input activation (x) and output activation (y).
        @return A tuple that contains (in order): number of correct outputs, fraction of correct outputs.
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
        self._v_biases = None
        self._v_weights = None

    def __del__(self):
        if self._workers is not None:
            self._workers.shutdown()
        
    def optimize(
        self,
        network: Network,
        num_epochs,
        training_set: Dataset,
        print_progress=False):
        """
        @param network The network to train.
        """
        start_time = timer()

        self._prepare_param_velocities(network)
        for _ in range(num_epochs):
            # Prepare mini batch data
            training_set.shuffle()
            mini_batches = [
                training_set[bi : bi + self._mini_batch_size]
                for bi in range(0, len(training_set), self._mini_batch_size)]
            
            if print_progress:
                self._print_progress(0)

            # Train with mini batches
            for bi, mini_batch_data in enumerate(mini_batches):
                self._mini_batch_param_update(mini_batch_data, network, len(training_set))

                if print_progress:
                    fraction_done = (bi + 1) / len(mini_batches)
                    seconds_spent = timedelta(seconds=(timer() - start_time)).total_seconds()
                    ms_per_batch = seconds_spent / (bi + 1) * 1000
                    mins_left = seconds_spent / 60 / fraction_done * (1 - fraction_done)
                    self._print_progress(
                        fraction_done,
                        suffix=f" ({ms_per_batch:10.2f} ms/batch, {mins_left:7.2f} mins left)")

            self._total_epochs += 1

        self._train_time = timedelta(seconds=(timer() - start_time))
        self._total_train_time += self._train_time

    def total_cost(self, dataset: Dataset, network: Network):
        y_hats = _feedforward_dataset(dataset, network, False, self._workers, self._num_workers)

        cost = 0.0
        rcp_dataset_n = 1.0 / len(dataset)
        for y, y_hat in zip(dataset.ys, y_hats):
            cost += network.cost.eval(y_hat, vec.vector_2d(y)) * rcp_dataset_n

        # L2 regularization term
        sum_w2 = sum(np.linalg.norm(w) ** 2 for w in network.weights)
        cost += self._lambba * rcp_dataset_n * 0.5 * sum_w2

        return cost
    
    def performance(self, dataset: Dataset, network: Network):
        y_hats = _feedforward_dataset(dataset, network, False, self._workers, self._num_workers)
        results = [(np.argmax(y_hat), np.argmax(y)) for y, y_hat in zip(dataset.ys, y_hats)]
        
        num_right_answers = sum(1 for yi_hat, yi in results if yi_hat == yi)
        num_data = len(dataset)
        return (num_right_answers, num_right_answers / num_data)

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

    def _mini_batch_param_update(self, mini_batch_data, network: Network, n):
        """
        @param n Number of training samples. To see why dividing by `n` is used for regularization,
        see https://datascience.stackexchange.com/questions/57271/why-do-we-divide-the-regularization-term-by-the-number-of-examples-in-regularize.
        """
        del_bs, del_ws = _sgd_mini_batch_backpropagation(
            mini_batch_data, network, self._gradient_clip_norm, self._workers)

        # Update momentum parameters
        rcp_n = com.REAL_TYPE(1.0 / n)
        self._v_biases = [
            vb * self._momentum - self._eta * bi
            for vb, bi in zip(self._v_biases, del_bs)]
        self._v_weights = [
            vw * self._momentum - self._eta * self._lambba * rcp_n * w - self._eta * wi
            for w, vw, wi in zip(network.weights, self._v_weights, del_ws)]

        # Compute new biases and weights
        new_biases = [b + vb for b, vb in zip(network.biases, self._v_biases)]
        new_weights = [w + vw for w, vw in zip(network.weights, self._v_weights)]

        # Update biases and weights
        for layer, new_b, new_w in zip(network.hidden_layers, new_biases, new_weights):
            layer.update_params(new_b, new_w)

    def _prepare_param_velocities(self, network: Network):
        """
        Initialize gradient records if not already exist.
        """
        if self._v_biases is None:
            self._v_biases = [vec.zeros_from(b) for b in network.biases]

        if self._v_weights is None:
            self._v_weights = [vec.zeros_from(w) for w in network.weights]

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
    mini_batch_set: Dataset,
    network: Network,
    gradient_clip_norm,
    workers: futures.Executor=None):
    """
    Performs backpropagation for a collection of training data.
    @param mini_batch_set A chunk of training set for the mini batch.
    @param workers If not `None`, use the specified workers to perform the calculation.
    """
    rcp_mini_batch_n = com.REAL_TYPE(1.0 / len(mini_batch_set))

    del_bs = [vec.zeros_from(layer.bias) for layer in network.hidden_layers]
    del_ws = [vec.zeros_from(layer.weight) for layer in network.hidden_layers]

    # Approximating the true `del_b` and `del_w` from m samples for each hidden layer

    # Summation of gradients
    if workers is None:
        for x, y in zip(mini_batch_set.xs, mini_batch_set.ys):
            delta_del_bs, delta_del_ws = _sgd_backpropagation(x, y, network, gradient_clip_norm)
            del_bs = [bi + dbi for bi, dbi in zip(del_bs, delta_del_bs)]
            del_ws = [wi + dwi for wi, dwi in zip(del_ws, delta_del_ws)]
    else:
        backpropagation_args = (
            mini_batch_set.xs,
            mini_batch_set.ys,
            [network] * len(mini_batch_set),
            [gradient_clip_norm] * len(mini_batch_set))
        
        for result in workers.map(_sgd_backpropagation, *backpropagation_args):
            delta_del_bs, delta_del_ws = result
            del_bs = [bi + dbi for bi, dbi in zip(del_bs, delta_del_bs)]
            del_ws = [wi + dwi for wi, dwi in zip(del_ws, delta_del_ws)]

    # Averaging over m samples
    del_bs = [bi * rcp_mini_batch_n for bi in del_bs]
    del_ws = [wi * rcp_mini_batch_n for wi in del_ws]

    return (del_bs, del_ws)

def _feedforward_dataset(
    dataset: Dataset,
    network: Network,
    is_training,
    workers: futures.Executor=None,
    num_workers=0,
    batch_size=10):
    """
    Feedforward an entire dataset.
    @param is_training `True` if the intent is to train the network, `False` for inference.
    """
    # TODO: apply batch size to layers

    y_hats = np.empty(dataset.ys.shape)
    if workers is None:
        for xi, x in enumerate(dataset.xs):
            y_hats[xi] = network.feedforward(vec.vector_2d(x), is_training)
    else:
        assert num_workers != 0, f"requires `num_workers` for dividing the dataset"
        sub_datasets = dataset.split(num_workers)
        
        args = (
            sub_datasets,
            [network] * num_workers,
            [is_training] * num_workers)
        
        yi_begin = 0
        for sub_y_hats in workers.map(_feedforward_dataset, *args):
            yi_end = yi_begin + len(sub_y_hats)
            y_hats[yi_begin : yi_end] = sub_y_hats
            yi_begin = yi_end
        assert yi_begin == len(y_hats)

    return y_hats

def _gradient_clip(delta, gradient_clip_norm):
    # Potentially apply gradient clipping
    if gradient_clip_norm < np.finfo(com.REAL_TYPE).max:
        delta_norm = np.linalg.norm(delta)
        if delta_norm >= gradient_clip_norm:
            delta = delta / delta_norm * gradient_clip_norm
    return delta
