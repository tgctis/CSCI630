import numpy as np

x = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])

y = np.array([[0, 0, 0, 1]]).T

synaptic_weights = np.random.random((2, 1))

for iteration in range(10):
    z = np.dot(x, synaptic_weights)
    sigmoid = 1.0/(1.0 + np.exp(-z))
    error = (y - sigmoid)
    sigmoidDerivative = sigmoid * (1.0 - sigmoid)
    synaptic_weights += np.dot(x.T, (error*sigmoidDerivative))

print synaptic_weights