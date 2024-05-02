import idx_file
import common as com
from model.network import Network
from model.layer import FullyConnected, Convolution, Pool
from model.layer_wrapper import Reshape
from model.activation import Sigmoid, Softmax, ReLU, Tanh

import numpy as np

import random
import os


def label_to_output(training_label):
    output = np.zeros((1, 10, 1), dtype=np.float32)
    output[0, int(training_label), 0] = 1
    return output


def train_basic_network():
    training_inputs = idx_file.load("./dataset/mnist/train-images-idx3-ubyte.gz").astype(np.float32)
    training_labels = idx_file.load("./dataset/mnist/train-labels-idx1-ubyte.gz").astype(np.float32)
    test_inputs = idx_file.load("./dataset/mnist/t10k-images-idx3-ubyte.gz").astype(np.float32)
    test_labels = idx_file.load("./dataset/mnist/t10k-labels-idx1-ubyte.gz").astype(np.float32)

    # Normalize to [0, 1]
    training_inputs = training_inputs / np.float32(255)
    test_inputs = test_inputs / np.float32(255)

    image_shape = (1, training_inputs.shape[1], training_inputs.shape[2])
    training_data = [
        (np.reshape(image, image_shape), label_to_output(label))
        for image, label in zip(training_inputs, training_labels)]
    test_data = [
        (np.reshape(image, image_shape), label_to_output(label))
        for image, label in zip(test_inputs, test_labels)]

    random.Random(6942).shuffle(training_data)
    validation_data = training_data[50000:]
    training_data = training_data[:50000]

    # network = Network([num_image_pixels, 30, 10])
    # network = Network(
    #     [FullyConnected(num_image_pixels, 100), FullyConnected(100, 10)])
    fc1 = FullyConnected(image_shape, (1, 10, 10), activation=Tanh())
    cov1 = Convolution(fc1.output_shape, (5, 5), 4)
    mp1 = Pool(cov1.output_shape, (4, 1, 1), com.PoolingMode.MAX)
    rs1 = Reshape(mp1, output_shape=(1, mp1.output_shape[-2] * mp1.output_shape[-3], mp1.output_shape[-1]))
    fc2 = FullyConnected(rs1.output_shape, (1, 10, 1), activation=Softmax())
    network = Network([fc1, cov1, rs1, fc2])
    # cov1 = Convolution(image_shape, (5, 5), 4)
    # mp1 = Pool(cov1.output_shape, (1, 2, 2), com.PoolingMode.MAX)
    # fc1 = FullyConnected(mp1.output_shape, (1, 10, 10), activation=Tanh())
    # fc2 = FullyConnected(, (1, 10, 1), activation=Softmax())
    # network = Network([fc1, cov1, mp1, fc2])
    # network = Network([num_image_pixels, 10])
    # network.stochastic_gradient_descent(training_data, 30, 10, eta=0.5, test_data=test_data)
    network.stochastic_gradient_descent(
        training_data,
        30,
        10,
        eta=0.05,
        lambba=1,
        eval_data=validation_data,
        report_eval_performance=True,
        report_eval_cost=True,
        num_workers=os.cpu_count())
    # network.stochastic_gradient_descent(training_data, 60, 10, eta=0.1, momentum=0.0, lambba=5.0, test_data=test_data)
    # network.stochastic_gradient_descent(training_data, 30, 10, eta=0.1, momentum=0.5, lambba=5.0, test_data=test_data)
    # network.stochastic_gradient_descent(training_data, 1000, 10, eta=0.1, momentum=0.2, lambba=5.0, test_data=test_data)


if __name__ == '__main__':
    train_basic_network()
