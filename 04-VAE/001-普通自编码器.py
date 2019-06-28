"""

@file   : 001-普通自编码器.py

@author : xiaolu

@time   : 2019-06-28

"""
# 极简的编码器
from keras.datasets import mnist
from keras.layers import *
from keras.models import Model
import matplotlib.pyplot as plt


def ae(x_train):
    encode_dim = 2
    batch_size = 64

    input = Input(shape=(28*28,))
    # 编码
    dense1 = Dense(encode_dim, activation='relu')(input)

    # 解码
    dense2 = Dense(28*28, activation='relu')(dense1)

    autoencoder = Model(inputs=input, outputs=dense2)

    encoder = Model(inputs=input, outputs=dense1)

    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    autoencoder.fit(x_train, x_train, batch_size=batch_size, epochs=1)

    return autoencoder, encoder


def plot_representation(encode_images, y_test):
    # 画隐层的显示
    plt.scatter(encode_images[:, 0], encode_images[:, 1], c=y_test, s=3)
    plt.colorbar()
    plt.show()


def show_images(decode_images, x_test):
    # params: dencode_images 是经过编码-解码过程后的图像
    # params: x_test 使我们原始的图像
    # 我们各画九张
    plt.gray()
    for i in range(1, 10):
        plt.subplot(2, 9, i)
        plt.imshow(decode_images[i].reshape((28, 28)))

        plt.subplot(2, 9, i+9)
        plt.imshow(x_test[i].reshape((28, 28)))
    plt.show()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 28*28)).astype('float32') / 255.
    x_test = x_test.reshape((-1, 28*28)).astype('float32') / 255.

    autoencoder, encoder = ae(x_train)

    # 保存模型
    autoencoder.save('ae1.h5')

    # 对测试集进行编码
    encode_images = encoder.predict(x_test)
    plot_representation(encode_images, y_test)

    # 画测试图像
    decode_images = autoencoder.predict(x_test)
    show_images(decode_images, x_test)








