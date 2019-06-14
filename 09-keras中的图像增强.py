"""

@file   : 09-keras中的图像增强.py

@author : xiaolu

@time   : 2019-06-14

"""
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
import matplotlib.pyplot as plt
import os

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 1.原始的图片先画9张
for i in range(9):
    plt.subplot("33{}".format(i+1))
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
plt.show()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))


# 2.通过ImageDateGenerator进行中心化，标准化
# 图像特征标准化
# imgGen = ImageDataGenerator(featurewise_center=True,   # 中心化
#                             featurewise_std_normalization=True,  # 标准化
#                             )
# imgGen.fit(x_train)
# for x_batch, y_batch in imgGen.flow(x_train, y_train, batch_size=9):
#     for i in range(0, 9):
#         plt.subplot("33{}".format(i+1))
#         plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
#     plt.show()
#     break


# 3.ZCA白化  图像的白化处理是线性代数操作，能够减少图像像素矩阵的冗余和相关性
# imgGen = ImageDataGenerator(zca_whitening=True)
# imgGen.fit(x_train)
# for x_batch, y_batch in imgGen.flow(x_train, y_train, batch_size=9):
#     for i in range(0, 9):
#         plt.subplot("33{}".format(i+1))
#         plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
#     plt.show()
#     break


# 4.图像旋转
# imgGen = ImageDataGenerator(rotation_range=90)  # 旋转90度
# imgGen.fit(x_train)
# for x_batch, y_batch in imgGen.flow(x_train, y_train, batch_size=9):
#     for i in range(0, 9):
#         plt.subplot("33{}".format(i+1))
#         plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
#     plt.show()
#     break


# 5.图像移动
# imgGen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)  # 宽随意平移20%以内  高同理
# imgGen.fit(x_train)
# for x_batch, y_batch in imgGen.flow(x_train, y_train, batch_size=9):
#     for i in range(0, 9):
#         plt.subplot("33{}".format(i+1))
#         plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
#     plt.show()
#     break


# 6.图像剪切
# imgGen = ImageDataGenerator(shear_range=0.2)  # 浮点数，剪切强度（逆时针方向的剪切变换角度）。是用来进行剪切变换的程度。
# imgGen.fit(x_train)
# for x_batch, y_batch in imgGen.flow(x_train, y_train, batch_size=9):
#     for i in range(0, 9):
#         plt.subplot("33{}".format(i + 1))
#         plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
#     plt.show()
#     break


# 7.图像翻转
# imgGen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)  # 进行随机水平翻转  随机竖直翻转
# imgGen.fit(x_train)
# for x_batch, y_batch in imgGen.flow(x_train, y_train, batch_size=9):
#     for i in range(0, 9):
#         plt.subplot("33{}".format(i + 1))
#         plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
#     plt.show()
#     break


# 保存增强后的图像  我们就保存图像翻转后的图片
# 创建目录
imgGen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)  # 进行随机水平翻转  随机竖直翻转
imgGen.fit(x_train)
try:
    os.mkdir('image')
except:
    print("目录已经存在")
for x_batch, y_batch in imgGen.flow(x_train, y_train, batch_size=9,
                                    save_to_dir='image',
                                    save_prefix='x',
                                    save_format='png'):
    for i in range(0, 9):
        plt.subplot("33{}".format(i + 1))
        plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()
    break

