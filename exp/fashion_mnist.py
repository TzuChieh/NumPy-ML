import common as com
from dataset import idx_file, Dataset
from model.network import Network
from model.optimizer import SGD, Adam
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

def load_data(batch_size):
    """
    Loads Fashion-MNIST dataset. The loaded data is in exactly the same format as MNIST. Though the images are
    labeled in [0, 9], they actually have different meanings than MNIST. Here is what each number means in
    the Fashion-MNIST dataset:
    
    Label 	Description
    -------------------
    0       T-shirt/top
    1       Trouser
    2       Pullover
    3       Dress
    4       Coat
    5       Sandal
    6       Shirt
    7       Sneaker
    8       Bag
    9       Ankle boot
    """
    training_images = idx_file.load("./dataset/fashion_mnist/train-images-idx3-ubyte.gz")
    training_labels = idx_file.load("./dataset/fashion_mnist/train-labels-idx1-ubyte.gz")
    test_images = idx_file.load("./dataset/fashion_mnist/t10k-images-idx3-ubyte.gz")
    test_labels = idx_file.load("./dataset/fashion_mnist/t10k-labels-idx1-ubyte.gz")

    # Normalize to [0, 1]
    training_images = training_images.astype(com.REAL_TYPE) / com.REAL_TYPE(255)
    test_images = test_images.astype(com.REAL_TYPE) / com.REAL_TYPE(255)

    training_set = Dataset(images_to_inputs(training_images), labels_to_outputs(training_labels), batch_size=batch_size)
    test_set = Dataset(images_to_inputs(test_images), labels_to_outputs(test_labels), batch_size=batch_size)
    
    return (training_set, test_set)

def load_basic_network_preset():
    training_set, test_set = load_data(batch_size=10)

    training_set.shuffle(6942)
    validation_set = training_set[50000:]
    training_set = training_set[:50000]

    fc1 = FullyConnected(training_set.input_shape, (1, 10, 10), activation=Sigmoid())
    fc2 = FullyConnected(fc1.output_shape, (1, 10, 1), activation=Softmax())
    network = Network([fc1, fc2])

    optimizer = SGD(
        eta=0.05,
        lambba=1,
        num_workers=1)

    preset = TrainingPreset()
    preset.name = "Fashion-MNIST Basic Network"
    preset.network = network
    preset.optimizer = optimizer
    preset.training_set = training_set
    preset.validation_set = validation_set
    preset.test_set = test_set
    preset.num_epochs = 10

    return preset

def load_network_preset():
    training_set, test_set = load_data(batch_size=20)

    training_set.shuffle(6942)
    validation_set = training_set[50000:]
    training_set = training_set[:50000]

    cov1 = Convolution(training_set.input_shape, (4, 4), 16, activation=ReLU(), use_tied_bias=False)
    cov2 = Convolution(cov1.output_shape, (3, 3), 12, activation=ReLU(), use_tied_bias=False)
    rs1 = Reshape(cov2.output_shape, (1, cov2.output_shape[-2] * cov2.output_shape[-3], cov2.output_shape[-1]))
    fc1 = FullyConnected(rs1.output_shape, (1, 100, 1), activation=Tanh())
    fc2 = FullyConnected(fc1.output_shape, (1, 10, 1), activation=Softmax())
    network = Network([cov1, cov2, rs1, fc1, fc2])

    # optimizer = SGD(
    #     eta=0.008,
    #     momentum=0.9,
    #     lambba=1,
    #     num_workers=os.cpu_count())
    
    optimizer = Adam(
        lambba=1,
        num_workers=os.cpu_count())

    preset = TrainingPreset()
    preset.name = "Fashion-MNIST Network"
    preset.network = network
    preset.optimizer = optimizer
    preset.training_set = training_set
    preset.validation_set = validation_set
    preset.test_set = test_set
    # preset.num_epochs = 30
    preset.num_epochs = 1000

    return preset

def load_deeper_network_preset():
    training_set, test_set = load_data(batch_size=20)

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

    optimizer = SGD(
        eta=0.04,
        lambba=1,
        num_workers=os.cpu_count())

    preset = TrainingPreset()
    preset.name = "Fashion-MNIST Deeper Network"
    preset.network = network
    preset.optimizer = optimizer
    preset.training_set = training_set
    preset.validation_set = validation_set
    preset.test_set = test_set
    preset.num_epochs = 60

    return preset
