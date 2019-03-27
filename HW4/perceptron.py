import numpy as np

texas = 1
florida = 0
# cutoff = 30, I cut off the values at 30, and assigned them 1's & 0's if they were above/below & = 30.
# x = np.array([[11, 70], [35, 11], [21, 45], [60, 80], [37, 32], [26, 64], [44, 30], [12, 60]])
x = np.array([[0, 1], [1, 0], [0, 1], [1, 1], [1, 1], [0, 1]])  # , [1, 0], [0, 1]

y = np.array([[1, 0, 1, 0, 0, 1]]).T  # , 0, 1

synaptic_weights = np.random.random((2, 1))

for iteration in range(10):
    z = np.dot(x, synaptic_weights)
    sigmoid = 1.0/(1.0 + np.exp(-z))
    error = (y - sigmoid)
    sigmoidDerivative = sigmoid * (1.0 - sigmoid)
    synaptic_weights += np.dot(x.T, (error*sigmoidDerivative))


def convert_input(x):
    if x > 30:
        return 1
    return 0


test_x1 = 44
test_x2 = 30

val_x1 = convert_input(test_x1)
val_x2 = convert_input(test_x2)

print("Consider the following data [" + str(test_x1) + ", " + str(test_x2)
      + "] => [ " + str(val_x1) + "," + str(val_x2) + "]")
newZ = np.dot(np.array([val_x1, val_x2]), synaptic_weights)
activationOutput = 1.0/(1.0+np.exp(-newZ))
print("%.5f" % activationOutput)
