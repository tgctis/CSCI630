import math
import sys

# vars to use
ys = [7, 5, 8, 4, 7, 11, 100]


# mean of the passed in array
def mean(ys):
    tmp = 0
    for y in ys:
        tmp = tmp + y
    return tmp / len(ys)


# median of passed in array
def median(ys):
    ys.sort()
    i = int(math.ceil(len(ys)/2))
    return ys[i]


# calculates the L1 loss
def lone(ys, guess):
    n = len(ys)
    sigma = 0
    for y in ys:
        sigma = sigma + abs(y - guess)
    return sigma / n


# calculates the L2 loss
def ltwo(ys, guess):
    n = len(ys)
    sigma = 0
    for y in ys:
        sigma = sigma + math.pow((y - guess), 2)
    return sigma/n


# runs the trials to a stop point to find the best L1 & L2 values
def runtrial(ys, top):
    best = sys.maxint
    best_guess = 0
    for guess in range(1, top):
        # This will keep the best guess and the best L1 value
        l1 = lone(ys, guess)
        if l1 < best:
            best_guess = guess
            best = l1

    print "Best L1 guess = %s, L1 = %.2f" % (best_guess, lone(ys, best_guess))
    print "Median = %d" % median(ys)

    best = sys.maxint
    best_guess = 0
    for guess in range(1, top):
        # This will keep the best guess and the best L2 value
        l2 = ltwo(ys, guess)
        if l2 < best:
            best_guess = guess
            best = l2

    print "Best L2 guess = %s, L2 = %.2f" % (best_guess, ltwo(ys, best_guess))
    print "Mean = %d" % mean(ys)


runtrial(ys, 100)
