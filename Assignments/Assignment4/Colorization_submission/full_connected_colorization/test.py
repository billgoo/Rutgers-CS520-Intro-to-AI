import xlrd
data = xlrd.open_workbook('dict.xls')
dict = {}
# print(len(data.sheets()))
# table = data.sheets()[0]
# print(len(table.col_values(0)))
# print(table.col_values(0))
for i in range(len(data.sheets())):
    table = data.sheets()[i]
    print(i)
    for j in range(len(table.col_values(0))):
        data_list = []
        for ele in table.row_values(j):
            if ele == '':
                break
            data_list.append(ele)
        if len(data_list) == 88:
            t = tuple(data_list[0: 4])
            li = data_list[4:]
            dict[t] = []
            dict[t].append(li)
        elif len(data_list) == 114:
            t = tuple(data_list[0: 6])
            li = data_list[6:]
            dict[t] = []
            dict[t].append(li)
        else:
            t = tuple(data_list[0: 9])
            li = data_list[9:]
            dict[t] = []
            dict[t].append(li)
print(len(dict))
# data_list = []
# for ele in table.row_values(1):
#     if ele == '':
#         break
#     data_list.append(ele)
# print(len(data_list))
# print(data_list)
# t = tuple([1, 2, 6, 2])
# t1 = tuple([2, 3, 6])
# dict[t] = []
# dict[t1] = []
# dict[t].append([111, 444, 555])
# dict[t1].append([123, 445])
# for key, value in dict.items():
#     for i in range(len(key)):
#         print(key[i])
#     for j in range(len(value)):
#         print(value[j])
