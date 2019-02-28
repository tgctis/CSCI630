import math
import sys


actions = ['r', 'f', '!']
max_factorial = 170
max_iter = 100000


def rotorooter(x, _sequence):
    sequence = list(_sequence)[:]
    sequence.reverse()
    while len(sequence) > 0:
        action = sequence.pop()
        if action == 'r':
            # print "Trying sqrt: x = " + str(x)
            x = math.sqrt(x)
            # time.sleep(sleeper)
        elif action == 'f':
            # print "Trying floor: x = " + str(x)
            x = float(x)
            x = math.floor(x)
            # time.sleep(sleeper)
        elif action == '!':
            # print "Trying factorial: x = " + str(x)
            if 0 < x <= max_factorial:
                x = math.factorial(int(x))
            else:
                # print "Max factorial reached..."
                return -1
            # time.sleep(sleeper)
        else:
            print "Bad input : " + action
            return -1
    return x


class Tree:
    def __init__(self, cargo, left=None, right=None, visited=False):
        self.cargo = cargo
        self.left = left
        self.right = right
        self.visited = visited

    def __str__(self):
        return str(self.cargo)


def print_tree(tree, depth = 0):
    if tree == None:
        return
    print "Depth: " + str(depth) + "\tRoot: " + str(tree) + "\tLeft: " + str(tree.left) + "\tRight: " + str(tree.right)
    print_tree(tree.left, depth + 1)
    print_tree(tree.right, depth + 1)


def buildTree(k, tree):
    left = Tree('!')
    right = Tree('r')

    if k == 0:
        return tree
    else:
        tree.left = left
        buildTree(k - 1, tree.left)
        tree.right = right
        buildTree(k - 1, tree.right)
        return tree


def printPaths(root):
    path = []
    return printPathsRec(root, path, 0, [])


def printPathsRec(root, path, pathLen, seqs):
    # base case
    if root is None:
        return

    if len(path) > pathLen:
        path[pathLen] = root.cargo
    else:
        path.append(root.cargo)

    pathLen += 1

    if root.left is None and root.right is None:
        seqs.append(printArray(path))
    else:
        printPathsRec(root.left, path, pathLen, seqs)
        printPathsRec(root.right, path, pathLen, seqs)

    return seqs


def printArray(seq):
    return ''.join(seq)


def factroot(n, seq='', x = 4):
    n = int(n)
    iterativeAddition = 3
    iterateCount = 0
    ITER_MAX = 6
    found = False
    intlist = []

    # print "Finding " + str(n)
    depth = 2
    while not found and iterateCount < ITER_MAX:
        depth = 2 + (iterativeAddition * iterateCount)
        # print "Iteration #" + str(iterateCount) + ". Depth: " + str(depth)
        tree = buildTree(depth, Tree('f'))
        # print_tree(tree)
        seqs = printPaths(tree)
        # print len(seqs)
        # print "Possible sequences: " + str(seqs)

        for i in range(0, len(seqs)-1):
            result = rotorooter(x, seqs[i])
            if math.floor(result) < sys.maxint:
                intlist.insert(int(math.floor(result)), [math.floor(result), str(seq) + seqs[i] + "f"])
            if result - 1 < n < result + 1 and result > n:
                result = math.floor(result)
                seq = str(seq) + str(seqs[i])
                print "Tried sequence #" + str(i) + " : " + str(4) + str(seq) + "f = " + str(result)
                found = True
                break
        iterateCount += 1

    if not found:
        # print "Not found in " + str(ITER_MAX) + " iterative deepenings with x = "+str(x)+". Max decision tree depth = " + str(depth)
        return intlist
    else:
        return []


def run(n):
    ret = factroot(n)
    triedx = [1, 4]
    MAX_ITER = 20
    runnum = 1
    while len(ret) > 0 and ret[-1] != n and runnum < MAX_ITER:
        for x in ret:
            _x = x[0]
            _seq = x[1]
            if _x not in triedx and _x > 0:
                # print x
                _ret = factroot(n, _seq, _x)
                triedx.append(_x)
                if len(ret) == 0:
                    break
                else:
                    runnum += 1
                    ret = ret + _ret


if len(sys.argv) > 1:
    run(sys.argv[1])
else:
    for i in range(1, 50):
        run(i)


# print rotorooter(4, 'f!r!r!!rrrrr!r!rrrf')