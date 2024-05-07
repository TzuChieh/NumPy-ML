import dataset.idx_file as data
import common as com
from model.network import Network
from model.optimizer import StochasticGradientDescent
from model.layer import FullyConnected, Convolution, Pool, Reshape, Dropout
from model.activation import Sigmoid, Softmax, ReLU, Tanh
from model.preset import TrainingPreset

import numpy as np

import random
import os


def label_to_output(label):
    output = np.zeros((1, 10, 1), dtype=np.float32)
    output[0, int(label), 0] = 1
    return output

def load_data():
    training_inputs = data.load("./dataset/mnist/train-images-idx3-ubyte.gz").astype(np.float32)
    training_labels = data.load("./dataset/mnist/train-labels-idx1-ubyte.gz").astype(np.float32)
    test_inputs = data.load("./dataset/mnist/t10k-images-idx3-ubyte.gz").astype(np.float32)
    test_labels = data.load("./dataset/mnist/t10k-labels-idx1-ubyte.gz").astype(np.float32)

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
    
    return (training_data, test_data)

def load_basic_network_preset():
    training_data, test_data = load_data()

    random.Random(6942).shuffle(training_data)
    validation_data = training_data[50000:]
    training_data = training_data[:50000]

    image_shape = training_data[0][0].shape

    fc1 = FullyConnected(image_shape, (1, 10, 10), activation=Tanh())
    fc2 = FullyConnected(fc1.output_shape, (1, 10, 1), activation=Softmax())
    network = Network([fc1, fc2])

    optimizer = StochasticGradientDescent(
        10,
        eta=0.05,
        lambba=1,
        num_workers=8)

    preset = TrainingPreset()
    preset.name = "MNIST Basic Network"
    preset.network = network
    preset.optimizer = optimizer
    preset.training_data = training_data
    preset.validation_data = validation_data
    preset.test_data = test_data
    preset.num_epochs = 30

    return preset

def load_network_preset():
    training_data, test_data = load_data()

    random.Random(6942).shuffle(training_data)
    validation_data = training_data[50000:]
    training_data = training_data[:50000]

    image_shape = training_data[0][0].shape

    cov1 = Convolution(image_shape, (5, 5), 20)
    mp1 = Pool(cov1.output_shape, (1, 2, 2), com.PoolingMode.MAX)
    rs1 = Reshape(mp1.output_shape, (1, mp1.output_shape[-2] * mp1.output_shape[-3], mp1.output_shape[-1]))
    d1 = Dropout(rs1.output_shape, 0.5)
    fc1 = FullyConnected(d1.output_shape, (1, 100, 1), activation=Tanh())
    fc2 = FullyConnected(fc1.output_shape, (1, 10, 1), activation=Softmax())
    network = Network([cov1, mp1, rs1, d1, fc1, fc2])

    optimizer = StochasticGradientDescent(
        20,
        eta=0.05,
        lambba=1,
        num_workers=os.cpu_count())

    preset = TrainingPreset()
    preset.name = "MNIST Network"
    preset.network = network
    preset.optimizer = optimizer
    preset.training_data = training_data
    preset.validation_data = validation_data
    preset.test_data = test_data
    preset.num_epochs = 30

    return preset

def load_deeper_network_preset():
    training_data, test_data = load_data()

    random.Random(6942).shuffle(training_data)
    validation_data = training_data[50000:]
    training_data = training_data[:50000]

    image_shape = training_data[0][0].shape

    cov1 = Convolution(image_shape, (5, 5), 20)
    mp1 = Pool(cov1.output_shape, (1, 2, 2), com.PoolingMode.MAX)
    cov2 = Convolution(mp1.output_shape, (5, 5), 40)
    mp2 = Pool(cov2.output_shape, (1, 2, 2), com.PoolingMode.MAX)
    rs1 = Reshape(mp2.output_shape, (1, mp2.output_shape[-2] * mp2.output_shape[-3], mp2.output_shape[-1]))
    d1 = Dropout(rs1.output_shape, 0.5)
    fc1 = FullyConnected(d1.output_shape, (1, 100, 1), activation=Tanh())
    fc2 = FullyConnected(fc1.output_shape, (1, 10, 1), activation=Softmax())
    network = Network([cov1, mp1, cov2, mp2, rs1, d1, fc1, fc2])

    optimizer = StochasticGradientDescent(
        20,
        eta=0.04,
        lambba=1,
        num_workers=os.cpu_count())

    preset = TrainingPreset()
    preset.name = "MNIST Network"
    preset.network = network
    preset.optimizer = optimizer
    preset.training_data = training_data
    preset.validation_data = validation_data
    preset.test_data = test_data
    preset.num_epochs = 40

    return preset
