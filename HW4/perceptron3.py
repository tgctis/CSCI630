import random

texas = 1
florida = 0

# Learning rate
alpha = 0.6

# Data
xs = [[11, 70], [35, 11], [21, 45], [60, 80], [37, 32], [26, 64], [44, 30], [12, 60]]
ys = [1, 0, 1, 0, 0, 1, 0, 1]
l = len(xs)

# Weights
ws = [0.1, 0.9]
# ws = [random.random(), random.random()]


# step activation function
def g(_sum):
    if _sum/l > 1:
        return 1
    else:
        return 0


# each learning iteration
for step in range(10):
    for i in range(0, l):
        _sum = 0
        # Summation over all the x values & weights
        for j in range(0, len(xs[i])):
            _sum += ws[j] * xs[i][j]

        # calculated y
        _y = g(_sum)

        # check for accuracy
        if _y > ys[i]:
            error = -1
        elif _y < ys[i]:
            error = 1
        else:
            error = 0

        # back propagate here
        for j in range(0, len(xs[i])):
            ws[j] += ws[j] * alpha * error


# Does the final check of the data to see how the weights worked.
def check():
    total = l
    right = 0
    for _i in range(l):
        test_x1 = xs[_i][0]
        test_x2 = xs[_i][1]
        actual = ys[_i]
        computed = g(test_x1 * ws[0] + test_x2 * ws[1])
        if computed == actual:
            right += 1
        print("Consider the following data [" + str(test_x1) + ", " + str(test_x2) + "] = " + str(actual)
              + " Computed: " + str(computed))
    print "Total accuracy: %.2f%%" % ((right * 1.0/total * 1.0) * 100)


check()
print "Final weights: ", ws
