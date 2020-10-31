import math
import random
import generate
import numpy
import copy
import queue

d = {1: 'flat', 2: 'hilly', 3: 'forested', 4: 'caves'}


# search the target
def search(obj, cell, c_belief, dim, rule, que):
    board = obj.get('board')
    row = obj.get('target').row
    col = obj.get('target').col
    p_belief = copy.deepcopy(c_belief)
    if row == cell[0] and col == cell[1]:
        landscape = obj.get('board')[row][col]
        if landscape == 1 and in_flat():
            print("True")
            rtn = {'out': True, 'next': []}
            return rtn
        elif landscape == 2 and in_hill():
            print("True")
            rtn = {'out': True, 'next': []}
            return rtn
        elif landscape == 3 and in_forest():
            print("True")
            rtn = {'out': True, 'next': []}
            return rtn
        elif landscape == 4 and in_cave():
            print("True")
            rtn = {'out': True, 'next': []}
            return rtn
        else:
            c_belief = move(obj, dim, p_belief, c_belief)
            # print("update belief:\n", c_belief)
            if rule == 1:
                next_search = rule_1(p_belief, c_belief, dim, que)
            elif rule == 2:
                next_search = rule_2(board, p_belief, c_belief, dim, que)
            # print('Target at: ', [obj.get('target').row, obj.get('target').col], '\tNext search: ', next_search)
            rtn = {'out': False, 'next': next_search}
            return rtn
    else:
        # print("False")
        c_belief = move(obj, dim, p_belief, c_belief)
        # print("update belief:\n", c_belief)
        if rule == 1:
            next_search = rule_1(p_belief, c_belief, dim, que)
        elif rule == 2:
            next_search = rule_2(board, p_belief, c_belief, dim, que)
        # print('Target at: ', [obj.get('target').row, obj.get('target').col], '\tNext search: ', next_search)
        rtn = {'out': False, 'next': next_search}
        return rtn


# search with constraint
def constraint_search(obj, cell, dim, c_belief, que):
    board = obj.get('board')
    row = obj.get('target').row
    col = obj.get('target').col
    p_belief = copy.deepcopy(c_belief)
    # print(c_belief)
    if row == cell[0] and col == cell[1]:
        landscape = obj.get('board')[row][col]
        if landscape == 1 and in_flat():
            print("True")
            rtn = {'out': True, 'next': []}
            return rtn
        elif landscape == 2 and in_hill():
            print("True")
            rtn = {'out': True, 'next': []}
            return rtn
        elif landscape == 3 and in_forest():
            print("True")
            rtn = {'out': True, 'next': []}
            return rtn
        elif landscape == 4 and in_cave():
            print("True")
            rtn = {'out': True, 'next': []}
            return rtn
        else:
            c_belief = move(obj, dim, p_belief, c_belief)
            # print("update belief:\n", c_belief)
            next_search = get_next(board, p_belief, c_belief, cell, dim, que)
            # print('Target at: ', [obj.get('target').row, obj.get('target').col], '\tNext search: ', next_search)
            rtn = {'out': False, 'next': next_search}
            return rtn
    else:
        # print("False")
        c_belief = move(obj, dim, p_belief, c_belief)
        # print("update belief:\n", c_belief)
        next_search = get_next(board, p_belief, c_belief, cell, dim, que)
        # print('Target at: ', [obj.get('target').row, obj.get('target').col], '\tNext search: ', next_search)
        rtn = {'out': False, 'next': next_search}
        return rtn


# search the cell with highest probability containing the target
def rule_1(p_belief, c_belief, dim, que):
    change = False
    for i in range(dim):
        if change:
            break
        for j in range(dim):
            if int(c_belief[i][j] * 1000000) != int(p_belief[i][j] * 1000000):
                change = True
                break
    if change:
        que.queue.clear()
        for i in range(dim):
            for j in range(dim):
                p = c_belief[i][j] - p_belief[i][j]
                que.put((-p, [i, j]))
        return que.get()[1]
    else:
        # for i in range(dim):
        #     for j in range(dim):
        #         p = c_belief[i][j]
        #         que.put((-p, [i, j]))
        return que.get()[1]


# search the cell with highest probability finding the target
def rule_2(board, p_belief, c_belief, dim, que):
    change = False
    for i in range(dim):
        for j in range(dim):
            if int(c_belief[i][j] * 1000000) != int(p_belief[i][j] * 1000000):
                change = True
                break
        if change:
            break
    if change:
        que.queue.clear()
        max_pro = -1
        landscape_que = queue.PriorityQueue()
        for i in range(dim):
            for j in range(dim):
                p = c_belief[i][j] - p_belief[i][j]
                if p > max_pro:
                    max_pro = p
                que.put((-p, [i, j]))
        next_ele = que.get()
        while -next_ele[0] == max_pro:
            i = next_ele[1][0]
            j = next_ele[1][1]
            landscape_que.put((board[i][j], next_ele[1]))
            next_ele = que.get()
        return landscape_que.get()[1]
    else:
        # for i in range(dim):
        #     for j in range(dim):
        #         que.put((board[i][j], [i, j]))
        return que.get()[1]


# determine next cell with constraint
def get_next(board, p_belief, c_belief, cell, dim, que):
    change = False
    for i in range(dim):
        for j in range(dim):
            if int(c_belief[i][j] * 1000000) != int(p_belief[i][j] * 1000000):
                change = True
                break
        if change:
            break
    if change:
        que.queue.clear()
        for i in range(dim):
            for j in range(dim):
                dis = abs(i - cell[0]) + abs(j - cell[1])
                if dis > 0:
                    que.put((dis, [i, j]))
        closest = que.get()
        if closest[0] == 1:
            return closest[1]
        else:
            que.queue.clear()
            if closest[1][0] > cell[0]:
                que.put((board[cell[0] - 1][cell[1]], [cell[0] - 1, cell[1]]))
                if closest[1][1] > cell[1]:
                    que.put((board[cell[0]][cell[1] + 1], [cell[0], cell[1] + 1]))
                else:
                    que.put((board[cell[0]][cell[1] - 1], [cell[0], cell[1] - 1]))
            elif closest[1][0] < cell[0]:
                que.put((board[cell[0] - 1][cell[1]], [cell[0] - 1, cell[1]]))
                if closest[1][1] > cell[1]:
                    que.put((board[cell[0]][cell[1] + 1], [cell[0], cell[1] + 1]))
                else:
                    que.put((board[cell[0]][cell[1] - 1], [cell[0], cell[1] - 1]))
            elif closest[1][0] == cell[0]:
                if closest[1][1] - cell[1] > 0:
                    que.put((board[cell[0]][cell[1] + 1], [cell[0], cell[1] + 1]))
                else:
                    que.put((board[cell[0]][cell[1] - 1], [cell[0], cell[1] - 1]))
            elif closest[1][1] == cell[1]:
                if closest[1][0] - cell[0] > 0:
                    que.put((board[cell[0] + 1][cell[1]], [cell[0] + 1, cell[1]]))
                else:
                    que.put((board[cell[0] - 1][cell[1]], [cell[0] - 1, cell[1]]))
            return que.get()[1]
    else:
        return que.get()[1]


# belief matrix
def generate_belief(dim):
    matrix = numpy.zeros((dim, dim), float)
    belief = 1 / math.pow(dim, 2)
    for i in range(dim):
        for j in range(dim):
            matrix[i][j] = belief
    # print("previous belief:\n", matrix)
    return matrix


# If target in flat
def in_flat():
    p = random.random()
    if p > 0.1:
        return True
    return False


# If target in hill
def in_hill():
    p = random.random()
    if p > 0.3:
        return True
    return False


# If target in forest
def in_forest():
    p = random.random()
    if p > 0.7:
        return True
    return False


# If target in cave
def in_cave():
    p = random.random()
    if p > 0.9:
        return True
    return False


# target move
def move(obj, dim, p_belief, c_belief):
    board = obj.get('board')
    r = obj.get('target').row
    c = obj.get('target').col
    type1 = board[r][c]
    obj.get('target').move(dim)
    r = obj.get('target').row
    c = obj.get('target').col
    type2 = board[r][c]
    # print(d.get(type1) + "-" + d.get(type2))
    c_belief = update_belief(board, dim, p_belief, c_belief, type1, type2)
    return c_belief


# update current belief
def update_belief(board, dim, p_belief, c_belief, type1, type2):
    remain = 0
    for i in range(dim):
        for j in range(dim):
            if board[i][j] != type1 and board[i][j] != type2:
                c_belief[i][j] = 0
            else:
                valid = True
                if board[i][j] == type1:
                    valid = check_neighbor(board, i, j, type2)
                elif board[i][j] == type2:
                    valid = check_neighbor(board, i, j, type1)
                if not valid:
                    c_belief[i][j] = 0
                else:
                    remain += 1
                    c_belief[i][j] = 1
    for i in range(dim):
        for j in range(dim):
            if c_belief[i][j] == 1:
                c_belief[i][j] = 1 / remain
    return c_belief


# Mark the cell that doesn't have the target
def check_neighbor(board, i, j, type):
    if i == 0 and j == 0:
        if board[i + 1][j] == type or board[i][j + 1] == type:
            return True
    elif i == 0 and j == len(board[0]) - 1:
        if board[i + 1][j] == type or board[i][j - 1] == type:
            return True
    elif i == len(board[0]) - 1 and j == 0:
        if board[i - 1][j] == type or board[i][j + 1] == type:
            return True
    elif i == len(board[0]) - 1 and j == len(board[0]) - 1:
        if board[i - 1][j] == type or board[i][j - 1] == type:
            return True
    elif i == 0:
        if board[i][j - 1] == type or board[i][j + 1] == type or board[i + 1][j] == type:
            return True
    elif i == len(board[0]) - 1:
        if board[i][j - 1] == type or board[i][j + 1] == type or board[i - 1][j] == type:
            return True
    elif j == 0:
        if board[i - 1][j] == type or board[i + 1][j] == type or board[i][j + 1] == type:
            return True
    elif j == len(board[0]) - 1:
        if board[i - 1][j] == type or board[i + 1][j] == type or board[i][j - 1] == type:
            return True
    else:
        if board[i + 1][j] == type or board[i][j + 1] == type or board[i - 1][j] == type or board[i][j - 1] == type:
            return True
    return False


# num = 5
# new_obj_1 = generate.generate(num)
# b = new_obj_1.get('board')
# print(b)
# curr_belief = generate_belief(num)
# search_que_1 = queue.PriorityQueue()
# ori_row_1 = 0
# ori_col_1 = 0
# count_1 = 0
# out = False
# print("Rule 1: ")
# while not out:
#     count_1 += 1
#     rtn_obj = search(new_obj_1, [ori_row_1, ori_col_1], curr_belief, num, 1, search_que_1)
#     out = rtn_obj.get('out')
#     next_cell = rtn_obj.get('next')
#     if len(next_cell) > 0:
#         ori_row_1 = rtn_obj.get('next')[0]
#         ori_col_1 = rtn_obj.get('next')[1]
# print("Search times: ", count_1)
# print("\n")
#
# num = 5
# new_obj_1 = generate.generate(num)
# curr_belief = generate_belief(num)
# search_que_1 = queue.PriorityQueue()
# ori_row_1 = 0
# ori_col_1 = 0
# count_1 = 0
# out = False
# print("Rule 2: ")
# while not out:
#     count_1 += 1
#     rtn_obj = search(new_obj_1, [ori_row_1, ori_col_1], curr_belief, num, 2, search_que_1)
#     out = rtn_obj.get('out')
#     next_cell = rtn_obj.get('next')
#     if len(next_cell) > 0:
#         ori_row_1 = rtn_obj.get('next')[0]
#         ori_col_1 = rtn_obj.get('next')[1]
# print("Search times: ", count_1)
# print("\n")
#
# print("Constraint search: ")
# out = False
# search_que_3 = queue.PriorityQueue()
# count_3 = 0
# ori_row_3 = 0
# ori_col_3 = 0
# while not out:
#     count_3 += 1
#     rtn_obj = constraint_search(new_obj_1, [ori_row_3, ori_col_3], num, curr_belief, search_que_3)
#     out = rtn_obj.get('out')
#     next_cell = rtn_obj.get('next')
#     if len(next_cell) > 0:
#         ori_row_3 = rtn_obj.get('next')[0]
#         ori_col_3 = rtn_obj.get('next')[1]
# print("Search times: ", count_3)
# print("\n")