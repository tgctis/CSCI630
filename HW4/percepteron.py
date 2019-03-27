import random
import numpy as np
texas = 1
florida = 0
x = np.array([
      [11, 70]
    , [35, 11]
    , [21, 45]
    , [60, 80]
    , [37, 32]
    , [26, 64]
    , [44, 30]
    # , [12, 60]
])
y = np.array([
    [1, 0, 1, 0, 0, 1, 0]#, [1], [0], [1]
]).T

synaptic_weights = np.random.random((2, 1))
# print synaptic_weights

for iteration in range(10):
    z = np.dot(x, synaptic_weights)
    print z
    sigmoid = 1.0/(1.0 + np.exp(-z))
    print sigmoid
    error = (y-sigmoid)
    # print error
    sigmoidDerivative = sigmoid * (1.0 - sigmoid)
    # print sigmoidDerivative
    synaptic_weights += np.dot(x.T, (error*sigmoidDerivative))
    # print synaptic_weights
    # synaptic_weights += error * sigmoidDerivative
    # print synaptic_weights
# print synaptic_weights
#
# print("Consider the following data [44, 30]")
# newZ = np.dot(np.array([44, 30]), synaptic_weights)
# activationOutput = 1.0/(1.0+np.exp(-newZ))
# print("%.5f" % activationOutput)
