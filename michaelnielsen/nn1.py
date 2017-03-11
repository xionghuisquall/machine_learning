import random
import numpy as np
import mnist_loader

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

        act = x

        zs = []
        zds = []
        acts = [x]
        for (b, w) in zip(self.biases, self.weights):
            z = np.dot(w, act) + b
            zs.append(z)

            act = sigmoid(z)
            acts.append(act)

            zd = sigmoid_prime(z)
            zds.append(zd)

        delta = (acts[-1] - y) * zds[-1]
        # thetas = [theta_n]

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, acts[-2].transpose())

        for l in xrange(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * zds[-l]
            # thetas[-l] = theta

            nabla_b[-l] = delta
            tmp = np.dot(delta, acts[-l - 1].transpose())
            nabla_w[-l] = tmp


        return (nabla_b, nabla_w)

    def update_mini_batch(self, batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for (x, y) in batch:
            b, w = self.backprop(x, y)
            nabla_b = [nb + db for (nb, db) in zip(nabla_b, b)]
            nabla_w = [nw + dw for (nw, dw) in zip(nabla_w, w)]

        self.weights = [w - (eta / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def SGD(self, training_data, epochs, mini_batch_size, eta, testing_data = None):
        if testing_data:
            n_test = len(testing_data)

        n = len(training_data)
        for i in xrange(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k : k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if testing_data:
                print "Epoch {0}: {1} / {2}".format(i,self.evaluate(testing_data), n_test)
            else:
                print "Epoch {0} completed".format(i)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def cost_derivate(output_activation, y):
    return output_activation - y



def run():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10])
    # net.SGD(training_data, 30, 10, 3.0, test_data)
    net.SGD(training_data, 1, 1, 3.0, test_data)


if __name__ == "__main__":
    run()