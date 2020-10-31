import moving_target
import generate
import copy
import queue
from xlwt import *

file = Workbook(encoding='utf-8')
table = file.add_sheet('dim=50')
table.write(0, 0, "rule 1")
table.write(1, 0, "rule 2")
table.write(2, 0, "rule 3")

num = 50
new_obj_1 = generate.generate(num)
new_obj_2 = copy.deepcopy(new_obj_1)
new_obj_3 = copy.deepcopy(new_obj_2)
target_row = new_obj_1.get('target').row
target_col = new_obj_1.get('target').col
b = new_obj_1.get('board')
print(b)
curr_belief = moving_target.generate_belief(num)
curr_belief_1 = copy.deepcopy(curr_belief)
curr_belief_2 = copy.deepcopy(curr_belief)
curr_belief_3 = copy.deepcopy(curr_belief)

run = 1
average_1 = 0
average_2 = 0
average_3 = 0
while run <= 5:
    search_que_1 = queue.PriorityQueue()
    out = False
    ori_row_1 = 0
    ori_col_1 = 0
    count_1 = 0
    print("Rule 1: ")
    while not out:
        count_1 += 1
        rtn_obj = moving_target.search(new_obj_1, [ori_row_1, ori_col_1], curr_belief_1, num, 1, search_que_1)
        out = rtn_obj.get('out')
        next_cell = rtn_obj.get('next')
        if len(next_cell) > 0:
            ori_row_1 = rtn_obj.get('next')[0]
            ori_col_1 = rtn_obj.get('next')[1]
    average_1 += count_1
    table.write(0, run, count_1)
    print("Search times: ", count_1)
    print("\n")

    print("Rule 2: ")
    out = False
    search_que_2 = queue.PriorityQueue()
    for ri in range(num):
        for cj in range(num):
            search_que_2.put((b[ri][cj], [ri, cj]))
    count_2 = 0
    ori_row_2 = search_que_2.get()[1][0]
    ori_col_2 = search_que_2.get()[1][1]
    while not out:
        count_2 += 1
        rtn_obj = moving_target.search(new_obj_2, [ori_row_2, ori_col_2], curr_belief_2, num, 2, search_que_2)
        out = rtn_obj.get('out')
        next_cell = rtn_obj.get('next')
        if len(next_cell) > 0:
            ori_row_2 = rtn_obj.get('next')[0]
            ori_col_2 = rtn_obj.get('next')[1]
    average_2 += count_2
    table.write(1, run, count_2)
    print("Search times: ", count_2)
    print("\n")

    print("Constraint search: ")
    out = False
    search_que_3 = queue.PriorityQueue()
    count_3 = 0
    ori_row_3 = 0
    ori_col_3 = 0
    while not out:
        count_3 += 1
        rtn_obj = moving_target.constraint_search(new_obj_3, [ori_row_3, ori_col_3], num, curr_belief_3, search_que_3)
        out = rtn_obj.get('out')
        next_cell = rtn_obj.get('next')
        if len(next_cell) > 0:
            ori_row_3 = rtn_obj.get('next')[0]
            ori_col_3 = rtn_obj.get('next')[1]
    average_3 += count_3
    table.write(2, run, count_3)
    print("Search times: ", count_3)
    print("\n")

    run += 1
    new_obj_1.get('target').row = target_row
    new_obj_1.get('target').col = target_col
    new_obj_2.get('target').row = target_row
    new_obj_2.get('target').col = target_col
    new_obj_3.get('target').row = target_row
    new_obj_3.get('target').col = target_col
    curr_belief_1 = copy.deepcopy(curr_belief)
    curr_belief_2 = copy.deepcopy(curr_belief)
    curr_belief_3 = copy.deepcopy(curr_belief)

table.write(0, 11, average_1 / 5)
table.write(1, 11, average_2 / 5)
table.write(2, 11, average_3 / 5)
file.save("C:/Users/Jim_Lu/Desktop/Probability Search/test50_3.csv")
print("Average search times of rule 1: ", average_1 / 5)
print("Average search times of rule 2: ", average_2 / 5)
print("Average search times of rule 3: ", average_3 / 5)
