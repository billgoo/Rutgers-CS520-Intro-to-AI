# Yan (bill) Gu
# yg369
# 189001028
#
# Two A* approach

import numpy as np
from openheap import *


# find path use Euclidean Distance
def a_star_euclidean(grid):
    dictionary = a_star_search(grid, 0)
    dim_x = grid.shape[0]
    dim_y = grid.shape[1]
    g_score = dictionary['gScore']
    path = dictionary['PATH']
    fringe = 0
    if dictionary["havePath"]:
        # add the (0, 0)
        node_num = g_score[dim_x - 1, dim_y - 1] + 1
        fringe = len(dictionary['closeSet'])
    else:
        # do not have a path, infinite
        node_num = np.inf
        return {'PATH': [], 'NODE': node_num, 'FRINGE': fringe}
    return {'PATH': path, 'NODE': node_num, 'FRINGE': fringe}


# find path use Manhattan Distance
def a_star_manhattan(grid):
    dictionary = a_star_search(grid, 1)
    dim_x = grid.shape[0]
    dim_y = grid.shape[1]
    g_score = dictionary['gScore']
    path = dictionary['PATH']
    fringe = 0
    if dictionary["havePath"]:
        node_num = g_score[dim_x - 1, dim_y - 1]
        fringe = len(dictionary['closeSet'])
    else:
        # do not have a path, infinite
        node_num = np.inf
        return {'PATH': [], 'NODE': node_num, 'FRINGE': fringe}
    return {'PATH': path, 'NODE': node_num, 'FRINGE': fringe}


def reconstruct_path(came_from, current):
    # it is reverse path
    total_path = [list(current)]
    while current in came_from:
        current = came_from[current]
        total_path.append(list(current))
    # reverse the list
    total_path.reverse()
    return total_path


def a_star_search(grid, distance_sign):
    """Find the shortest path from start to goal.
    :param grid: The maze map in a numpy array.
    :param distance_sign: A sign to determine which distance to choose, Euclidean or Manhattan.
    :return: The function returns the best path found and numbers of nodes,
    and store it in a dictionary.
    """
    out_dict = dict()

    # get dim
    dim_x = grid.shape[0]
    dim_y = grid.shape[1]

    goal = [dim_x - 1, dim_y - 1]

    # distance from start node, initially unknown, set to infinite
    g_score = np.full((dim_x, dim_y), np.inf)
    # f = g + h, initially unknown, set to infinite
    f_score = np.full((dim_x, dim_y), np.inf)

    # distance from start to start
    g_score[0, 0] = 0
    f_score[0, 0] = heuristic([0, 0], goal, distance_sign)

    # The set of nodes already travels
    closed_set = []

    # currently discovered nodes
    # start node
    open_set = OpenHeap()
    open_set.build_heap([(f_score[0, 0], [0, 0])])

    # most efficient previous step
    came_from = dict()

    # dist_between(current, neighbor)
    dist_between = 1

    # open list is not empty
    while open_set.currentSize > 0:

        # current := the node in openSet having the lowest fScore
        # current = open_set.heapList[0][1]
        # remove current node
        # if there are some nodes have same f,
        # we priority choose the one have larger g
        current = tie_breaking(open_set, g_score)[1]
        if current == goal:
            closed_set.append(goal)
            out_dict['PATH'] = reconstruct_path(came_from, tuple(current))
            out_dict['gScore'] = g_score
            out_dict['havePath'] = True
            out_dict['closeSet'] = closed_set
            return out_dict

        closed_set.append(current)

        neighbors = find_neighbors(grid, dim_x, dim_y, current)
        for neighbor in neighbors:
            if neighbor in closed_set:
                # ignore the neighbor which is already evaluated
                continue
            # The distance from start to a neighbor.
            # The distance between function may vary as per the solution requirements
            tentative_g_score = g_score[current[0], current[1]] + dist_between

            if not open_set.contains(neighbor):
                h = heuristic(neighbor, goal, distance_sign)
                f = g_score[current[0], current[1]] + dist_between + h
                open_set.insert((f, neighbor))

            elif tentative_g_score >= g_score[neighbor[0], neighbor[1]]:
                continue  # this is not a better path

            # This path is the best until now.
            came_from[tuple(neighbor)] = tuple(current)
            g_score[neighbor[0], neighbor[1]] = tentative_g_score
            h = heuristic(neighbor, goal, distance_sign)
            f_score[neighbor[0], neighbor[1]] = g_score[neighbor[0], neighbor[1]] + h

    # print("Can not find a path, AStar search has failed")
    out_dict['PATH'] = closed_set
    out_dict['gScore'] = g_score
    out_dict['havePath'] = False
    out_dict['closeSet'] = closed_set
    return out_dict


# function for calculating Distance
def heuristic(current, goal, sign):
    if sign == 0:
        # function for calculating Euclidean Distance
        euclidean_heuristic = np.sqrt(np.square(goal[0] - current[0])
                                      + np.square(goal[1] - current[1]))
        return euclidean_heuristic
    elif sign == 1:
        # function for calculating Manhattan Distance
        manhattan_heuristic = np.sqrt(np.square(goal[0] - current[0])
                                      + np.square(goal[1] - current[1]))
        return manhattan_heuristic


# function for choose a better node
def tie_breaking(open_set, g):
    ties = [open_set.remove()]
    while open_set.currentSize > 0 and open_set.heapList[0][0] == ties[0][0]:
        ties.append(open_set.remove())
    if len(ties) == 1:
        return ties[0]
    else:
        # tie breaking strategy based on larger g
        max_g = 0
        current_node = 0
        for node in ties:
            if max_g < g[node[1][0], node[1][1]]:
                max_g = g[node[1][0], node[1][1]]
                # the node with largest g and to be traveled
                current_node = node
        # insert the other useless nodes back
        for item in ties:
            if item != current_node:
                open_set.insert(item)
        return current_node


# find nodes that are in grid and not a obstacle
def find_neighbors(grid, dim_x, dim_y, current):
    neighbors = []
    [x, y] = current
    if x + 1 in range(dim_x) and grid[x + 1][y] == 0:
        neighbors.append([x + 1, y])
    if x - 1 in range(dim_x) and grid[x - 1][y] == 0:
        neighbors.append([x - 1, y])
    if y + 1 in range(dim_y) and grid[x][y + 1] == 0:
        neighbors.append([x, y + 1])
    if y - 1 in range(dim_y) and grid[x][y - 1] == 0:
        neighbors.append([x, y - 1])
    return neighbors

'''
if __name__ == '__main__':
    maze = np.array([[0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 0],
                     [0, 0, 1, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0]])
    solution = a_star_search(maze, 1)
    a = solution['closeSet']
    nodes = len(a)
    if not a:
        print('empty list')
    else:
        print('fuck', len(a))
    print('The length of path is : ', len(solution['PATH']))
    print('The number of nodes explored is : ', solution['gScore'][5, 5])
    print(solution['closeSet'])
    print(solution['PATH'])
'''
