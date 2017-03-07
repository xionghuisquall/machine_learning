import random
import numpy as np

# stochastic gradient descent + mini-batch, backpropagation
class Network(object):

    def __init__(self, sizes):
        # assume the first layer is input layer
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # zip([1, 2, 3], [4, 5, 6], [7, 8, 9]) -> [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
        # np.random.randn(x, y) - > matrix(x * y)
        self.weights = [np.random.randn(y, x) for (x, y) in zip(sizes[:-1], sizes[1:])]



    def forward(self, a):
        for (b, w) in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        a = x

        zs = []
        zds = []
        acts = []
        for (b, w) in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            zs.append(z)

            act = sigmoid(z)
            acts.append(act)

            zd = sigmoid_prime(z)
            zds.append(zd)

        theta_n = (acts[-1] - y) * zds[-1]
        thetas = [theta_n]

        nabla_b[-1] = theta_n
        nabla_w[-1] = np.dot(theta_n, acts[-2].transpose())

        for l in xrange(2, self.num_layers):
            theta = np.dot(self.weights[-l + 1].transpose(), thetas[-l + 1]) * zds[-l]
            thetas[-l] = theta

            nabla_b[-l] = theta
            nabla_w[-l] = np.dot(theta, acts[-l - 1].transpose())


        return (nabla_b, nabla_w)



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def cost_derivate(output_activation, y):
    return output_activation - y



