#
import numpy as np

# backtrack to find the shortest path
def addpath(father, f, path, xCore, yCore):
    if father[f] != -1:
        addpath(father, father[f], path, xCore, yCore)
    path.append([xCore[f], yCore[f]])
    return

def bfs(maze):
    nodes = []  # nodes that traverse
    path = []  # shortest path
    father = []  # last node
    xCore = []  # record the x-core of the visited point
    yCore = []  # record the y-core of the visited point

    # the next possible step
    step = [
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1]
    ]
    # start_x = 0  # x-core of the start point
    # start_y = 0  # y-core of the start point
    end_x = len(maze) - 1  # x-core of the end point
    end_y = len(maze) - 1  # y-core of the end point
    visited = [[0 for col in range(len(maze))] for row in range(len(maze))]  # mark the visited point
    visited[0][0] = 1  #the first point is visited
    start = 0  #keep track of the current point
    end = 1  #keep track of the end of the list
    fringe = 0
    out_of_maze = 0  #1 means out of maze
    xCore.append(0)
    yCore.append(0)
    father.append(-1)

    while start < end:
        if out_of_maze == 1:
            break
        if fringe < end - start:
            fringe = end -start
        for i in range(len(step)):
            next_x = xCore[start] + step[i][0]  #the x-core of the next step
            next_y = yCore[start] + step[i][1]  #the y-core of the next step
            #out of bound
            if next_x < 0 or next_y < 0 or next_x > end_x or next_y > end_y:
                continue
            if maze[next_x][next_y] == 0 and visited[next_x][next_y] == 0:
                visited[next_x][next_y] = 1
                xCore.append(next_x)
                yCore.append(next_y)
                father.append(start)
                nodes.append([next_x, next_y])
                end += 1
            if next_x == end_x and next_y == end_y:
                out_of_maze = 1
                addpath(father, start, path, xCore, yCore)
                path.append([end_x,end_y])
                break
        start += 1
    solution = {'PATH': path, 'NODE': nodes, 'FRINGE': fringe}
    return solution


#test
'''
maze = [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 0, 1],
    [1, 0, 0, 0]
]
maze = np.array(maze)
a = bfs(maze)
if not a:
    print('empty list')
else:
    print('fuck', len(a))
print("The shortest path is: ", a['PATH'])
print("The length of the path is: ", len(a['PATH']))
print("The nodes that traverse: ", a['NODE'])
'''