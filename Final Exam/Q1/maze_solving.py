import numpy as np

maze = []
with open('Maze.txt', 'rt') as f:
    for line in f:
        temp = []
        for c in line:
            if c == 'G':
                temp.append(0)
                continue
            if c != '\n':
                temp.append(int(c))
        maze.append(np.array(temp))
maze = np.array(maze)
# print(maze)
x, y = maze.shape[0], maze.shape[1]
axis = []
for i in range(1, x):
    for j in range(1, y):
        if maze[i, j]:
            continue
        else:
            # print((i,j))
            temp = [maze[i - 1, j - 1], maze[i - 1, j], maze[i - 1, j + 1], maze[i, j - 1],
                    maze[i, j + 1], maze[i + 1, j - 1], maze[i + 1, j], maze[i + 1, j + 1]]
            # print(temp)
            if temp.count(1) == 5:
                axis.append([i, j])
print(axis)


def move_left(x, y):
    if maze[x, y - 1] == 0:
        return x, y -1
    else:
        return x, y


if __name__ == '__main__':
    num_five_cell = 0
    print(len(axis))
    for i in range(len(axis)):
        x, y = axis[i][0], axis[i][1]
        for k in range(2):
            x, y = move_left(x, y)
        if [x, y] in axis:
            num_five_cell += 1
    print(num_five_cell)
