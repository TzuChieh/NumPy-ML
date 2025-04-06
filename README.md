# NumPy-ML

This is my testground for the basics of machine learning. The algorithms in this project are all implemented in pure NumPy (see `./requirements.txt`) with Python 3, and contains many training presets for you to explore. To get a taste of what this project has to offer, let us start by training an AI for recognizing handwritten digits by following these steps:

```Shell
git clone https://github.com/TzuChieh/Simple-NumPy-ML.git
cd Simple-NumPy-ML
python example.py
```

This will start training a very basic model right away. From the console output, you should see something like this:

```Shell
Epoch 6 / 10:
SGD: [████████████████████████████████████████] 100.00% (10.75 ms/batch, 0.00 mins left) | eval perf: 9343 / 10000 (0.9343) | eval cost: 0.4105 | Δt_epoch: 0:00:16.809815 | Δt_report: 0:00:01.866717
Epoch 7 / 10:
SGD: [████████████____________________________]  31.09% (10.89 ms/batch, 0.20 mins left)
```

Let it run a while until epoch 10 is reached, the training will stop and outputs a trained model in `./output/MNIST Basic Network.model`. From the log you can see that this model has around 95% accuracy in recognizing handwritten digits.

## Features

Currently most of the features are for building neural networks. Some visualization utilities are also provided for analyzing generated data.

### Layers

* Reshape
* Fully Reshape (similar to Reshape, but as a layer wrapper)
* Fully Connected (also commonly known as dense layer)
* Convolution (arbitrary kernel shape and stride; supports both tied and untied biases)
* Pool (arbitrary kernel shape and stride; supports max and mean pooling)
* Dropout

All layers support the following initialization modes (if applicable): Zeros, Ones, Constant, Gaussian, LeCun, Xavier, Kaiming He.

Source code (layers): `./model/layer.py`
Source code (parameter initializers): `./model/initializer.py`

### Activation Functions

* Identity (simply pass through variables)
* Sigmoid
* Tanh
* Softmax
* ReLU
* Leaky ReLU

Source code: `./model/activation.py`

### Cost Functions

* Quadratic (also known as MSE, L2)
* Cross Entropy

Source code: `./model/cost.py`

### Optimizers

* Stochastic Gradient Descent (SGD, with momentum)
* Adaptive moment estimation (Adam)

All optimizers support:

* Mini-batch
* L2 Regularization
* Gradient Clipping (by norm)
* Multi-core synchronous parameter update
* Multi-core asynchronous parameter update
  - With gradient staleness compensation

Source code: `./model/optimizer.py`

### Visualization

* A simple GUI program for viewing training reports

Source code: `./gui/`

## Interesting Reads

This project is inspired by a collection of resources that I found useful on the Internet:

* Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
* Yani Ioannou's blog posts for deriving [single-layer backpropagation](https://blog.yani.ai/deltarule/) and [multi-layer backpropagation](https://blog.yani.ai/backpropagation/)
* Sargur N. Srihari's [course on deep learning](https://cedar.buffalo.edu/~srihari/CSE676/) has a nice derivation of matrix/jacobian based backpropagation, [course note](https://cedar.buffalo.edu/~srihari/CSE676/6.5.2%20Chain%20Rule.pdf) (a copy of it can be found in ./misc/6.5.2 Chain Rule.pdf)
* Pavithra Solai's article [Convolutions and Backpropagations](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c) is easy to understand for cases without stride
* Mayank Kaushik's article [Backpropagation for Convolution with Strides](https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710) and its [Part 2](https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-fb2f2efc4faa) are well written for general cases (with stride)
* Rae's video explanation [Episode 6 - Convolution Layer Backpropagation - Convolutional Neural Network from Scratch](https://www.youtube.com/watch?v=njlyOAiK_yE) has nice animations for understanding the topic
* S. Do.'s article [Vectorized convolution operation using NumPy](https://medium.com/latinxinai/vectorized-convolution-operation-using-numpy-b122fd52fba3) explains how to use Einstein summation convention for convolution perfectly
* Vincent Dumoulin and Francesco Visin's [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)
* Jason Brownlee's [A Gentle Introduction to Dropout for Regularizing Deep Neural Networks](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/) is an informative overview of dropout with lots of references
* [This excellent answer](https://datascience.stackexchange.com/questions/117082/how-can-i-implement-dropout-in-scikit-learn/117083#117083) by Conner explains the practical aspect of dropout clearly
* [This answer](https://stackoverflow.com/questions/42670274/how-to-calculate-fan-in-and-fan-out-in-xavier-initialization-for-neural-networks) by adityassrana clearly explains how to determine fan_in and fan_out when initializing layer weights (with illustration)
* Course notes of [CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)

## Some Notes

* *You **DO NOT** need a GPU to train your model.*
* I do not care the execution speed of the constructed model as the main purpose of this project is for me to understand the basics in the field. It is slow, but can still get the job done in a reasonable amount of time (for small networks).
* Currently convolution/correlation is implemented in the naive way (sliding a kernel across the matrix). Ideally, both the feedforward and backpropagation pass of convolutional layer can be implemented as matrix multiplications.

## Datasets

### MNIST

The MNIST dataset for training and evaluation of handwritten digits are obtained from [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/). This dataset is included in `./dataset/mnist/`.

### Fashion-MNIST

The Fashion-MNIST dataset contains small grayscale images of clothing and can be obtained from their [GitHub repository](https://github.com/zalandoresearch/fashion-mnist). This is a MNIST-like dataset that is designed to be a drop-in replacement of MNIST, with greater variety and harder to predict. This dataset is included in `./dataset/fashion_mnist/`.

### CIFAR-10

### CIFAR-100

## Additional Dependencies (optional)

As mentioned earlier, the only required third-party library is NumPy. Additional libraries can be installed to support more functionalities (see `./requirements_extra.txt`). To install all dependencies in one go, pick the requirement files of your choice and execute (using two files as an example)

```Shell
pip install -r requirements.txt -r requirements_extra.txt
```
