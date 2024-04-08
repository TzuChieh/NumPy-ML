import random

import numpy as np

import idx_file
import model


def label_to_output(training_label):
    output = np.zeros((10, 1))
    output[training_label] = 1.0
    return output

training_inputs = idx_file.load("./database/train-images-idx3-ubyte.gz")
training_labels = idx_file.load("./database/train-labels-idx1-ubyte.gz")
test_inputs = idx_file.load("./database/t10k-images-idx3-ubyte.gz")
test_labels = idx_file.load("./database/t10k-labels-idx1-ubyte.gz")

# Normalize to [0, 1]
training_inputs = training_inputs / 255.0
test_inputs = test_inputs / 255.0

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

# network = model.Network([num_image_pixels, 30, 10])
network = model.Network([num_image_pixels, 100, 10])
# network = model.Network([num_image_pixels, 10])
# network.stochastic_gradient_descent(training_data, 30, 10, 0.5, test_data=test_data)
network.stochastic_gradient_descent(training_data, 60, 10, 0.1, lambba=5.0, test_data=test_data)

