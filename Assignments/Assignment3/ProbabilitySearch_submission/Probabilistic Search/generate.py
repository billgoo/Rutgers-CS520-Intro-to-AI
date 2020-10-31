import numpy
import random

class target:
    def __init__(self, dim):
        self.row = int(random.random() * dim)
        self.col = int(random.random() * dim)

    def move(self, dim):
        r = random.random()
        if r < 0.25:  #move up
            if self.row > 0:
                self.row -= 1
            else:
                self.row += 1
        elif 0.25 <= r < 0.5:  #move right
            if self.col + 1 < dim:
                self.col += 1
            else:
                self.col -= 1
        elif 0.5 <= r <= 0.75:  #move down
            if self.row + 1 < dim:
                self.row += 1
            else:
                self.col -= 1
        elif 0.75 <= r < 1:  #move left
            if self.col > 0:
                self.col -= 1
            else:
                self.col += 1


def generate(dim):
    # each number represent different landscape
    flat = 1
    hilly = 2
    forested = 3
    caves = 4

    board = numpy.zeros((dim, dim), int)
    for i in range(dim):
        for j in range(dim):
            p = random.random() * 10
            if p > 0 and p < 2:
                board[i][j] = flat
            elif p >= 2 and p < 5:
                board[i][j] = hilly
            elif p >= 5 and p < 8:
                board[i][j] = forested
            else:
                board[i][j] = caves
    t = target(dim)
    d = {'board': board, 'target': t}
    return d

#test
# obj = generate(5)
# board = obj.get('board')
# row = obj.get('target').row
# col = obj.get('target').col
# print(board)
# print(row, col, board[row][col])