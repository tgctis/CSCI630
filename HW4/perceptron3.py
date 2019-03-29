import random

texas = 1
florida = 0

alpha = 0.0604

xs = [[11, 70], [35, 11], [21, 45], [60, 80], [37, 32], [26, 64], [44, 30], [12, 60]]
ys = [1, 0, 1, 0, 0, 1, 0, 1]
l = len(xs)
ws = [random.random(), random.random()]


def g(_sum):
    if _sum/l > 1:
        return 1
    else:
        return 0


for step in range(100):
    for i in range(0, l-1):
        _sum = 0
        for j in range(0, len(xs[i])-1):
            _sum += ws[j] * xs[i][j]
        _y = g(_sum)
        if _y > ys[i]:
            error = -1
        elif _y < ys[i]:
            error = 1
        else:
            error = 0

        # back propagate here
        for j in range(0, len(xs[i]) - 1):
            ws[j] += alpha * error


test_x1 = 37
test_x2 = 32

print("Consider the following data [" + str(test_x1) + ", " + str(test_x2) + "]")
print g(test_x1 * ws[0] + test_x2 * ws[1])
