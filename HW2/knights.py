import argparse
import random
import math


board = []
c_board = []
c_knights = []


def check_constraints(x, y):
    r = 0
    if x+1 <= size - 1:
        if y+2 <= size - 1:
            r = r + board[x + 1][y + 2]
        if y-2 >= 0:
            r = r + board[x + 1][y - 2]
    if x-1 >= 0:
        if y + 2 <= size - 1:
            r = r + board[x - 1][y + 2]
        if y - 2 >= 0:
            r = r + board[x - 1][y - 2]
    if x+2 <= size - 1:
        if y+1 <= size - 1:
            r = r + board[x + 2][y + 1]
        if y-1 >= 0:
            r = r + board[x + 2][y - 1]
    if x-2 >= 0:
        if y + 1 <= size - 1:
            r = r + board[x - 2][y + 1]
        if y - 1 >= 0:
            r = r + board[x - 2][y - 1]
    return r


def check_board():
    conflicts = 0
    del c_knights[:]
    for x in range(size):
        for y in range(size):
            hits = check_constraints(x, y)
            c_board[x][y] = hits
            if hits > 0:
                if board[x][y] == 1:
                    c_knights.append([x, y])
                    conflicts = conflicts + 1
    return conflicts


def check_knights():
    knights = 0
    for x in range(size):
        for y in range(size):
            if board[x][y] == 1:
                knights = knights + 1
    return knights


def reset_game():
    for x in range(size):
        for y in range(size):
            c_board[x][y] = 0
            board[x][y] = 0


# Get random conflicted knight, minimize it's conflicts, repeat ad nauseam
def action():
    if len(c_knights) < 1:
        return
    knight = c_knights[random.randint(0, len(c_knights))-1]
    knight_x = knight[0]
    knight_y = knight[1]
    min_current_hits = check_constraints(knight_x, knight_y)
    min_x = knight[0]
    min_y = knight[1]
    board[knight_x][knight_y] = 0

    for x in range(size):
        for y in range(size):
            if board[x][y] == 0 and check_constraints(x, y) < min_current_hits:
                min_current_hits = check_constraints(x, y)
                min_x = x
                min_y = y
    board[min_x][min_y] = 1


def init(max_steps, knights):
    knight = 0
    reset_game()

    # Initial loading of the knights
    for x in range(size):
        for y in range(size):
            if knight < knights:
                board[x][y] = 1
                knight = knight + 1

    # Find solution, fail if none found
    for i in range(max_steps):
        if check_board() == 0:
            return True
        else:
            action()
    return False


def print_board(_board):
    ret = ""
    for x in range(size):
        ret = ret + "\n"
        for y in range(size):
            ret = ret + "[ " + str(_board[x][y]) + " ]"
    print ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("board", help="Board size, one side (standard board = 8)", type=int)
    args = parser.parse_args()

    size = args.board

    board = [[0 for x in range(size)] for y in range(size)]
    c_board = [[0 for x in range(size)] for y in range(size)]

    for i in range(size*size, 0, -1):
        if init(1000, i):
            break

    print "Board: "
    print_board(board)

    print"Conflicts: "
    print_board(c_board)

    print "Conflicted Knights: " + str(len(c_knights))
    print "Placed Knights: " + str(check_knights())
    print "Optimal Knights: " + str(int(math.floor((size * size) / 2)))
