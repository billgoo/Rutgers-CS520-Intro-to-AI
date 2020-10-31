import numpy


# if there is no path, returned path is []
# return the first path found, not the shortest
def dfs(maze):
    dim = len(maze)
    path = [[0, 0]]
    nodes = 1
    fringe = 1
    next_step = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    visit = [[0 for row in range(dim)] for col in range(dim)]
    visit[0][0] = 1
    while len(path) > 0:
        finish = True
        for i in range(len(next_step)):
            x = path[-1][0]
            y = path[-1][1]
            x_next = x + next_step[i][0]
            y_next = y + next_step[i][1]

            if x_next < 0 or y_next < 0 or x_next > dim - 1 or y_next > dim - 1 or \
                    maze[x_next][y_next] == 1 or visit[x_next][y_next] == 1:
                continue
            path.append([x_next, y_next])
            if fringe < len(path):
                fringe = len(path)
            nodes += 1
            visit[x_next][y_next] = 1
            finish = False
            break

        if path[-1][0] == dim - 1 and path[-1][1] == dim - 1:
            break
        if finish:
            path.pop()
    return {'PATH': path, 'NODE': nodes, 'FRINGE': fringe}


