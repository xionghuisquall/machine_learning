import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # zip([1, 2, 3], [4, 5, 6], [7, 8, 9]) -> [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
        # np.random.randn(x, y) - > matrix(x * y)
        self.weights = [np.random.randn(y, x) for (x, y) in zip(sizes[:-1], sizes[1:])]



    def forward(self, a):
        for (b, w) in zip(self.biases, self.weights):
            a = self.sigmod(np.dot(w, a) + b)
        return a

    def backprop(self):

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def cost_derivate(output_activation, y):
    return output_activation - y



