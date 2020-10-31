import numpy as np
# import colorization
import process_img
from PIL import Image
import matplotlib.pyplot as plt
from xlwt import *


def NN(img, ori, weight1, weight2, times, rgb, loss=[]):
    # for col in range(img[0]):
    layer0 = img
    layer1 = sigmoid(np.dot(layer0, weight1))
    layer2 = sigmoid(np.dot(layer1, weight2))
    # print(layer0)
    # print(layer1)
    layer2_error = ori / 255 - layer2
    # print(layer2_error)
    d_weights2 = np.dot(layer1.T, (2 * layer2_error * desigmoid(layer2)))
    d_weights1 = np.dot(layer0.T, (np.dot(2 * layer2_error * desigmoid(layer2), weight2.T) * desigmoid(layer1)))
    weight1 += d_weights1
    weight2 += d_weights2
    # for i in range(3):
    #     loss.write(times, i, layer2[0][i] * 255)
    if times == 49:
        rgb.append(layer2 * 255)


def getSurround(matrix, i, j, list):
    if i == 0 and j == 0:
        list.append([matrix[i][j], matrix[i][j + 1], matrix[i + 1][j], matrix[i + 1][j + 1]])
    elif i == 0 and j == len(matrix[0]) - 1:
        list.append([matrix[i][j - 1], matrix[i][j], matrix[i + 1][j - 1], matrix[i + 1][j]])
    elif i == len(matrix) - 1 and j == 0:
        list.append([matrix[i - 1][j], matrix[i - 1][j + 1], matrix[i][j], matrix[i][j + 1]])
    elif i == len(matrix) - 1 and j == len(matrix[0]) - 1:
        list.append([matrix[i - 1][j - 1], matrix[i - 1][j], matrix[i][j - 1], matrix[i][j]])
    elif i == 0:
        list.append([matrix[i][j - 1], matrix[i][j], matrix[i][j + 1], matrix[i + 1][j - 1], matrix[i + 1][j], matrix[i + 1][j + 1]])
    elif i == len(matrix) - 1:
        list.append([matrix[i - 1][j - 1], matrix[i - 1][j], matrix[i - 1][j + 1], matrix[i][j - 1], matrix[i][j],
                     matrix[i][j + 1]])
    elif j == 0:
        list.append([matrix[i - 1][j], matrix[i - 1][j + 1], matrix[i][j], matrix[i][j + 1], matrix[i + 1][j],
                     matrix[i + 1][j + 1]])
    elif j == len(matrix[0]) - 1:
        list.append([matrix[i - 1][j - 1], matrix[i - 1][j], matrix[i][j - 1], matrix[i][j], matrix[i + 1][j - 1],
                     matrix[i + 1][j]])
    else:
        step = [-1, 0, 1]
        tmp = []
        for a in step:
            for b in step:
                tmp.append(matrix[i + a][j + b])
        list.append(tmp)


# sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# sigmoid derivative
def desigmoid(y):
    return y * (1.0 - y)


# cn = colorization.Colorization('cifar-10-batches-py/data_batch_1')
# obj = cn.rgb2gray()
obj = process_img.image2Matrix('testimg/cat.jpg')
X = obj.get('gray')
red = obj.get('red')
green = obj.get('green')
blue = obj.get('blue')
surround_list = []
rgb_list = []
origin = []
for a in range(len(X)):
    for b in range(len(X[0])):
        tmp = []
        tmp.append(red[a][b])
        tmp.append(green[a][b])
        tmp.append(blue[a][b])
        origin.append(tmp)

for i in range(len(X)):
    for j in range(len(X[0])):
        getSurround(X, i, j, surround_list)
# print(surround_list[0])
# print(np.array(surround_list[0]))
# print(np.array(surround_list[0]).reshape((1, 4)))
# print(origin[0])
# print(np.array(origin[0]))
# print(np.array(origin[0]).reshape((1, 3)))
# print(len(surround_list), len(origin))
# file = Workbook(encoding='utf-8')
# count = 0
for ele in range(len(surround_list)):
    # count += 1
    # loss = file.add_sheet('loss' + str(count))
    if len(surround_list[ele]) == 4:
        X_arr = np.array(surround_list[ele]).reshape((1, 4))
        y_arr = np.array(origin[ele]).reshape((1, 3))
        weight1 = 2 * np.random.random((4, 12)) - 1
        weight2 = 2 * np.random.random((12, 3)) - 1
        for times in range(50):
            NN(X_arr.astype(np.float64), y_arr.astype(np.float64), weight1, weight2, times, rgb_list, loss)
    elif len(surround_list[ele]) == 6:
        X_arr = np.array(surround_list[ele]).reshape((1, 6))
        y_arr = np.array(origin[ele]).reshape((1, 3))
        weight1 = 2 * np.random.random((6, 12)) - 1
        weight2 = 2 * np.random.random((12, 3)) - 1
        for times in range(50):
            NN(X_arr.astype(np.float64), y_arr.astype(np.float64), weight1, weight2, times, rgb_list, loss)
    else:
        X_arr = np.array(surround_list[ele]).reshape((1, 9))
        y_arr = np.array(origin[ele]).reshape((1, 3))
        weight1 = 2 * np.random.random((9, 12)) - 1
        weight2 = 2 * np.random.random((12, 3)) - 1
        for times in range(50):
            NN(X_arr.astype(np.float64), y_arr.astype(np.float64), weight1, weight2, times, rgb_list, loss)


# table = file.add_sheet('new_rgb')
# for x in range(1024):
#     for y in range(len(rgb_list[x][0])):
#         table.write(x, y, rgb_list[x][0][y])
# file.save('new_data_3.xls')
# print(rgb_list[0][0][0])
# print(rgb_list[1][0][0])
r = []
g = []
b = []
for index in range(len(rgb_list)):
    r.append(rgb_list[index][0][0])
    g.append(rgb_list[index][0][1])
    b.append(rgb_list[index][0][2])
# new_red = Image.fromarray(np.reshape(r, (32, 32)).astype(np.int)).convert('L')
# new_green = Image.fromarray(np.reshape(g, (32, 32)).astype(np.int)).convert('L')
# new_blue = Image.fromarray(np.reshape(b, (32, 32)).astype(np.int)).convert('L')
new_red = Image.fromarray(np.reshape(r, (224, 225)).astype(np.int)).convert('L')
new_green = Image.fromarray(np.reshape(g, (224, 225)).astype(np.int)).convert('L')
new_blue = Image.fromarray(np.reshape(b, (224, 225)).astype(np.int)).convert('L')
img = Image.merge("RGB", (new_red, new_green, new_blue))
plt.imshow(img)
plt.show()
