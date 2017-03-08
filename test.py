import random

# Third-party libraries
import numpy as np

sizes = [2, 3, 1]
weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
biases = [np.random.randn(y, 1) for y in sizes[1:]]
print(biases)
# print(weights)

nabla_b = [np.zeros(b.shape) for b in biases]

pb  = [b1 + 2 * b2 for (b1, b2) in zip(nabla_b, biases) ]

# print 2 * biases
# pb = nabla_b + 2 * biases

print pb

# print(zip(sizes[:-1], sizes[1:]))

# print np.random.randn(2,3,4)

# print zip(biases, weights)

# print weights[0]
# print weights[0].transpose()

# theta = np.dot(weights[0].transpose(), [1, 2, 4])

# print np.dot(theta, [1, 2, 4].transpose())