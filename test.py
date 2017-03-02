import random

# Third-party libraries
import numpy as np

sizes = [2, 3, 1]
weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
biases = [np.random.randn(y, 1) for y in sizes[1:]]
print(biases)
print(weights)

print(zip(sizes[:-1], sizes[1:]))

# print np.random.randn(2,3,4)

# print zip(biases, weights)

print weights[0]
print weights[0].transpose()
print np.dot(weights[0].transpose(), [1, 2, 4])