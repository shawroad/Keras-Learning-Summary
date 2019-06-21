"""

@file   : 11-将keras中mnist数据集写入本地.py

@author : xiaolu

@time   : 2019-06-21

"""
from keras.datasets import mnist
import os
import cv2
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


data = []
temp = []
for i in range(6000):
    dat = x_train[i]
    label = y_train[i]
    temp = [dat, label]
    data.append(temp)


a, b, c, d, e, f, g, h, i, j = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

for dat in data:
    if dat[1] == 0:
        name = '0_' + str(a)
        if not os.path.exists("./1"):
            os.mkdir('./1')
        cv2.imwrite('./1/{}.png'.format(name), dat[0])
        a += 1
    elif dat[1] == 1:
        name = '1_' + str(b)
        if not os.path.exists("./2"):
            os.mkdir('./2')
        cv2.imwrite('./2/{}.png'.format(name), dat[0])
        b += 1
    elif dat[1] == 2:
        name = '2_' + str(c)
        if not os.path.exists("./2"):
            os.mkdir('./2')
        cv2.imwrite('./2/{}.png'.format(name), dat[0])
        c += 1
    elif dat[1] == 3:
        name = '3_' + str(d)
        if not os.path.exists("./3"):
            os.mkdir('./3')
        cv2.imwrite('./3/{}.png'.format(name), dat[0])
        d += 1
    elif dat[1] == 4:
        name = '4_' + str(e)
        if not os.path.exists("./4"):
            os.mkdir('./4')
        cv2.imwrite('./4/{}.png'.format(name), dat[0])
        e += 1
    elif dat[1] == 5:
        name = '5_' + str(f)
        if not os.path.exists("./5"):
            os.mkdir('./5')
        cv2.imwrite('./5/{}.png'.format(name), dat[0])
        f += 1
    elif dat[1] == 6:
        name = '6_' + str(g)
        if not os.path.exists("./6"):
            os.mkdir('./6')
        cv2.imwrite('./6/{}.png'.format(name), dat[0])
        g += 1
    elif dat[1] == 7:
        name = '7_' + str(h)
        if not os.path.exists("./7"):
            os.mkdir('./7')
        cv2.imwrite('./7/{}.png'.format(name), dat[0])
        h += 1
    elif dat[1] == 8:
        name = '8_' + str(i)
        if not os.path.exists("./8"):
            os.mkdir('./8')
        cv2.imwrite('./8/{}.png'.format(name), dat[0])
        i += 1

    elif dat[1] == 9:
        name = '9_' + str(j)
        if not os.path.exists("./9"):
            os.mkdir('./9')
        cv2.imwrite('./9/{}.png'.format(name), dat[0])
        j += 1



