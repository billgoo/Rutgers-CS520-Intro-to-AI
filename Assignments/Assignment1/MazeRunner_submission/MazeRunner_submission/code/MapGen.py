import numpy
import random


def maze_generate(dim, p):
    maze = numpy.zeros((dim, dim), int)
    for i in range(dim):
        for j in range(dim):
            if random.random() < p:
                maze[i][j] = 1
    maze[0][0] = 0
    maze[dim - 1][dim - 1] = 0
    return maze

