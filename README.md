# MNIST-training

My testground for the basics of neural networks. Following a collection of resources that I found useful on the Internet:

* Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
* Yani Ioannou's blog posts for deriving [single-layer backpropagation](https://blog.yani.ai/deltarule/) and [multi-layer backpropagation](https://blog.yani.ai/backpropagation/)
* Sargur N. Srihari's [course on deep learning](https://cedar.buffalo.edu/~srihari/CSE676/) has a nice derivation of matrix/jacobian based backpropagation, [course note](https://cedar.buffalo.edu/~srihari/CSE676/6.5.2%20Chain%20Rule.pdf) (a copy of it can be found in ./misc/6.5.2 Chain Rule.pdf)
* Pavithra Solai's article [Convolutions and Backpropagations](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c) is easy to understand for cases without stride
* Mayank Kaushik's article [Backpropagation for Convolution with Strides](https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710) and its [Part 2](https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-fb2f2efc4faa) are well written for general cases (with stride)
* Rae's video explanation [Episode 6 - Convolution Layer Backpropagation - Convolutional Neural Network from Scratch](https://www.youtube.com/watch?v=njlyOAiK_yE) has nice animations for understanding the topic

The implementation is in Python 3. Only NumPy and Matplotlib is required (see `./requirements.txt`).

## Database

The dataset for training and evaluation are obtained from [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/).
