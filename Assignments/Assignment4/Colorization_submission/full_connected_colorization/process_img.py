from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def image2Matrix(file):
    img = Image.open(file)
    img = np.array(img)
    # print(img)
    # print(len(img))
    # print(len(img[0]))
    # print(img[0][0][0])
    gray_list = []
    red_list = []
    green_list = []
    blue_list = []
    for i in range(len(img)):
        for j in range(len(img[i])):
            gray_list.append(0.21 * img[i][j][0] + 0.72 * img[i][j][1] + 0.07 * img[i][j][2])
            red_list.append(img[i][j][0])
            green_list.append(img[i][j][1])
            blue_list.append(img[i][j][2])
    gray = np.reshape(gray_list, [224, 225]).astype(int)
    red = np.reshape(red_list, [224, 225])
    green = np.reshape(green_list, [224, 225])
    blue = np.reshape(blue_list, [224, 225])
    # print(len(gray))
    # plt.imshow(gray, cmap=plt.cm.gray)
    # plt.show()
    return {'gray': gray, 'red': red, 'green': green, 'blue': blue}


# filename = 'testimg/cat.jpg'
# image2Matrix(filename)