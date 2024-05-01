# NumPy-ML

This is my testground for the basics of machine learning. The implementation is in Python 3 and only NumPy is required (see `./requirements.txt`). This project is inspired by a collection of resources that I found useful on the Internet:

* Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
* Yani Ioannou's blog posts for deriving [single-layer backpropagation](https://blog.yani.ai/deltarule/) and [multi-layer backpropagation](https://blog.yani.ai/backpropagation/)
* Sargur N. Srihari's [course on deep learning](https://cedar.buffalo.edu/~srihari/CSE676/) has a nice derivation of matrix/jacobian based backpropagation, [course note](https://cedar.buffalo.edu/~srihari/CSE676/6.5.2%20Chain%20Rule.pdf) (a copy of it can be found in ./misc/6.5.2 Chain Rule.pdf)
* Pavithra Solai's article [Convolutions and Backpropagations](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c) is easy to understand for cases without stride
* Mayank Kaushik's article [Backpropagation for Convolution with Strides](https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710) and its [Part 2](https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-fb2f2efc4faa) are well written for general cases (with stride)
* Rae's video explanation [Episode 6 - Convolution Layer Backpropagation - Convolutional Neural Network from Scratch](https://www.youtube.com/watch?v=njlyOAiK_yE) has nice animations for understanding the topic
* S. Do.'s article [Vectorized convolution operation using NumPy](https://medium.com/latinxinai/vectorized-convolution-operation-using-numpy-b122fd52fba3) explains how to use Einstein summation convention for convolution perfectly
* Vincent Dumoulin and Francesco Visin's [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)

## Some Notes

* I do not care the execution speed of the constructed network as the main purpose of this project is for me to understand the basics in the field of machine learning. It is slow, but can still get the job done in a reasonable amount of time (for small networks).
* Currently convolution/correlation is implemented in the naive way (sliding a kernel across the matrix). Ideally, both the feedforward and backpropagation pass of convolutional layer can be implemented as matrix multiplications.

## Additional Dependencies

As mentioned earlier, the only required third-party library is Numpy. Additional libraries can be installed to support more functionalities (see `./requirement_extra.txt`). To install all dependencies in one go, pick the requirement files of your choice and execute (using two files as an example)

> `pip install -r requirement1.txt -r requirement2.txt`

## Database

The MNIST dataset for training and evaluation of handwritten digits are obtained from [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/).
