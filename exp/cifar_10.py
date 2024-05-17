import common as com
from dataset import cifar_10, Dataset
from model.network import Network
from model.optimizer import SGD
from model.layer import FullyConnected, Convolution, Pool, Reshape, Dropout
from model.activation import Sigmoid, Softmax, ReLU, Tanh
from model.preset import TrainingPreset

import numpy as np

import os
import pickle
from pathlib import Path


def images_to_inputs(images):
    inputs = images.reshape((len(images), 1, images.shape[1], images.shape[2]))
    return inputs

def labels_to_outputs(labels):
    outputs = np.zeros((len(labels), 1, 10, 1), dtype=com.REAL_TYPE)
    for label, output in zip(labels, outputs):
        output[0, int(label), 0] = 1
    return outputs

def load_data(batch_size):
    """
    Loads CIFAR-10 dataset. Will try to download them if not already exist. The format of the data is documented
    on https://www.cs.toronto.edu/~kriz/cifar.html.
    """
    batch1_file = Path("./dataset/cifar_10/cifar-10-batches-py/data_batch_1")
    if not batch1_file.is_file():
        cifar_10.download()

    with open(batch1_file, 'rb') as f:
        batch1_dict = pickle.load(f, encoding='bytes')

    print(batch1_dict.keys())

def load_basic_network_preset():
    load_data(batch_size=10)



def load_network_preset():
    pass

def load_deeper_network_preset():
    pass
