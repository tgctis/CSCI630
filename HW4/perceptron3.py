import random

texas = 1
florida = 0

alpha = 0.9

xs = [[11, 70], [35, 11], [21, 45], [60, 80], [37, 32], [26, 64], [44, 30], [12, 60]]
ys = [1, 0, 1, 0, 0, 1, 0, 1]
l = len(xs)
# ws = [random.random(), random.random()]
ws = [0.1, 0.2]


# step activation function
def g(_sum):
    if _sum/l > 1:
        return 1
    else:
        return 0


# each learning iteration
for step in range(10):
    for i in range(0, l-1):
        _sum = 0
        # Summation over all the x values & weights
        for j in range(0, len(xs[i])-1):
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
        for j in range(0, len(xs[i]) - 1):
            ws[j] += ws[j] * alpha * error


# test_x1 = 35
# test_x2 = 11
# print("Consider the following data [" + str(test_x1) + ", " + str(test_x2) + "] = "
#       + str(g(test_x1 * ws[0] + test_x2 * ws[1])))


def check():
    for _i in range(l-1):
        test_x1 = xs[_i][0]
        test_x2 = xs[_i][1]

        print("Consider the following data [" + str(test_x1) + ", " + str(test_x2) + "] = " + str(ys[_i])
              + " Computed: " + str(g(test_x1 * ws[0] + test_x2 * ws[1])))


check()
