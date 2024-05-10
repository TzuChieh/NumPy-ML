import common as com
from dataset import idx_file, Dataset
from model.network import Network
from model.optimizer import StochasticGradientDescent
from model.layer import FullyConnected, Convolution, Pool, Reshape, Dropout
from model.activation import Sigmoid, Softmax, ReLU, Tanh
from model.preset import TrainingPreset

import numpy as np

import os


def images_to_inputs(images):
    inputs = images.reshape((len(images), 1, images.shape[1], images.shape[2]))
    return inputs

def labels_to_outputs(labels):
    outputs = np.empty((len(labels), 1, 10, 1), dtype=com.REAL_TYPE)
    for label, output in zip(labels, outputs):
        output[0, int(label), 0] = 1
    return outputs

def load_data():
    """
    Loads MNIST dataset. The images are handwritten digits labeled in [0, 9].
    """
    training_images = idx_file.load("./dataset/mnist/train-images-idx3-ubyte.gz")
    training_labels = idx_file.load("./dataset/mnist/train-labels-idx1-ubyte.gz")
    test_images = idx_file.load("./dataset/mnist/t10k-images-idx3-ubyte.gz")
    test_labels = idx_file.load("./dataset/mnist/t10k-labels-idx1-ubyte.gz")

    # Normalize to [0, 1]
    training_images = training_images.astype(com.REAL_TYPE) / com.REAL_TYPE(255)
    test_images = test_images.astype(com.REAL_TYPE) / com.REAL_TYPE(255)

    training_set = Dataset(images_to_inputs(training_images), labels_to_outputs(training_labels))
    test_set = Dataset(images_to_inputs(test_images), labels_to_outputs(test_labels))
    
    return (training_set, test_set)

def load_basic_network_preset():
    training_set, test_set = load_data()

    training_set.shuffle(6942)
    validation_set = training_set[50000:]
    training_set = training_set[:50000]

    fc1 = FullyConnected(training_set.input_shape, (1, 10, 10), activation=Tanh())
    fc2 = FullyConnected(fc1.output_shape, (1, 10, 1), activation=Softmax())
    network = Network([fc1, fc2])

    optimizer = StochasticGradientDescent(
        10,
        eta=0.05,
        lambba=1,
        num_workers=1)

    preset = TrainingPreset()
    preset.name = "MNIST Basic Network"
    preset.network = network
    preset.optimizer = optimizer
    preset.training_set = training_set
    preset.validation_set = validation_set
    preset.test_set = test_set
    preset.num_epochs = 30

    return preset

def load_network_preset():
    training_set, test_set = load_data()

    training_set.shuffle(6942)
    validation_set = training_set[50000:]
    training_set = training_set[:50000]

    cov1 = Convolution(training_set.input_shape, (5, 5), 20, use_tied_bias=False)
    mp1 = Pool(cov1.output_shape, (1, 2, 2), com.EPooling.MAX)
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
    preset.training_set = training_set
    preset.validation_set = validation_set
    preset.test_set = test_set
    preset.num_epochs = 30

    return preset

def load_deeper_network_preset():
    training_set, test_set = load_data()

    training_set.shuffle(6942)
    validation_set = training_set[50000:]
    training_set = training_set[:50000]

    cov1 = Convolution(training_set.input_shape, (5, 5), 20)
    mp1 = Pool(cov1.output_shape, (1, 2, 2), com.EPooling.MAX)
    cov2 = Convolution(mp1.output_shape, (5, 5), 40)
    mp2 = Pool(cov2.output_shape, (1, 2, 2), com.EPooling.MAX)
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
    preset.name = "MNIST Deeper Network"
    preset.network = network
    preset.optimizer = optimizer
    preset.training_set = training_set
    preset.validation_set = validation_set
    preset.test_set = test_set
    preset.num_epochs = 60

    return preset
