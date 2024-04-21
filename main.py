import random

import numpy as np

import idx_file
from model import Network
from model import FullyConnected
from model import Sigmoid, Softmax, ReLU


def label_to_output(training_label):
    output = np.zeros((10, 1), dtype=np.float32)
    output[int(training_label)] = 1
    return output

# Disabling division by 0 error. 0/0 division error can also be ignored by setting `invalid='ignore'`, but we
# choose to enable it so we can get notified for potential numerical issues. 
np.seterr(divide='ignore')

training_inputs = idx_file.load("./database/train-images-idx3-ubyte.gz").astype(np.float32)
training_labels = idx_file.load("./database/train-labels-idx1-ubyte.gz").astype(np.float32)
test_inputs = idx_file.load("./database/t10k-images-idx3-ubyte.gz").astype(np.float32)
test_labels = idx_file.load("./database/t10k-labels-idx1-ubyte.gz").astype(np.float32)

# Normalize to [0, 1]
training_inputs = training_inputs / np.float32(255)
test_inputs = test_inputs / np.float32(255)

num_image_pixels = training_inputs.shape[1] * training_inputs.shape[2]

training_data = [
    (np.reshape(image, (num_image_pixels, 1)), label_to_output(label))
    for image, label in zip(training_inputs, training_labels)]
test_data = [
    (np.reshape(image, (num_image_pixels, 1)), label_to_output(label))
    for image, label in zip(test_inputs, test_labels)]

random.shuffle(training_data)
validation_data = training_data[50000:]
training_data = training_data[:50000]

# network = Network([num_image_pixels, 30, 10])
# network = Network(
#     [FullyConnected(num_image_pixels, 100), FullyConnected(100, 10)])
network = Network(
    [FullyConnected(num_image_pixels, 100, activation=ReLU()), FullyConnected(100, 10, activation=Softmax())])
# network = Network([num_image_pixels, 10])
# network.stochastic_gradient_descent(training_data, 30, 10, eta=0.5, test_data=test_data)
network.stochastic_gradient_descent(
    training_data, 30, 10, eta=0.05, lambba=1, test_data=test_data, report_test_performance=True, report_test_cost=True)
# network.stochastic_gradient_descent(training_data, 60, 10, eta=0.1, momentum=0.0, lambba=5.0, test_data=test_data)
# network.stochastic_gradient_descent(training_data, 30, 10, eta=0.1, momentum=0.5, lambba=5.0, test_data=test_data)
# network.stochastic_gradient_descent(training_data, 1000, 10, eta=0.1, momentum=0.2, lambba=5.0, test_data=test_data)

