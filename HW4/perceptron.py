import numpy as np

texas = 1
florida = 0

alpha = 0.604

x = np.array([[11, 70], [35, 11], [21, 45], [60, 80], [37, 32], [26, 64], [44, 30], [12, 60]])
y = np.array([[1, 0, 1, 0, 0, 1, 0, 1]]).T

synaptic_weights = np.random.random((2, 1))


def f_activation(_z):
    sum = 0
    for e in _z:

        if e > 30:
            sum += 1
        else:
            sum -= 1
    return sum/len(_z)


for iteration in range(1):
    z = np.dot(x, synaptic_weights)
    activation = f_activation(z)
    for _y in y:
        if activation == _y:
            continue

        error = (_y - activation)
        if activation > _y:
            error = -1
        else:
            error = 1

        synaptic_weights[0] += synaptic_weights[0] * alpha * error
        synaptic_weights[1] += synaptic_weights[1] * alpha * error


test_x1 = 44
test_x2 = 30

print("Consider the following data [" + str(test_x1) + ", " + str(test_x2) + "]")
newZ = np.dot(np.array([test_x1, test_x2]), synaptic_weights)
activationOutput = f_activation(newZ)
print("%.5f" % activationOutput)
