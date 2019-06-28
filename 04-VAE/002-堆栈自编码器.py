"""

@file   : 002-堆栈自编码器.py

@author : xiaolu

@time   : 2019-06-28

"""
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import *
from keras.datasets import mnist


def stack_ae(x_train):
    layers1_dim = 128
    layers2_dim = 64
    layers3_dim = 32
    encoder_out_dim = 2

    inputs = Input(shape=(28*28, ))

    # 编码过程
    layers1 = Dense(layers1_dim, activation='relu')(inputs)
    layers2 = Dense(layers2_dim, activation='relu')(layers1)
    layers3 = Dense(layers3_dim, activation='relu')(layers2)
    encoder_out = Dense(encoder_out_dim)(layers3)

    # 解码过程
    de_layers1 = Dense(layers3_dim, activation='relu')(encoder_out)
    de_layers2 = Dense(layers2_dim, activation='relu')(de_layers1)
    de_layers3 = Dense(layers1_dim, activation='relu')(de_layers2)
    de_output = Dense(28*28)(de_layers3)

    stackAE = Model(inputs=inputs, outputs=de_output)
    encoders = Model(inputs=inputs, outputs=encoder_out)

    stackAE.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    stackAE.fit(x_train, x_train, batch_size=64, epochs=5)

    return stackAE, encoders


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

    autoencoder, encoder = stack_ae(x_train)

    # 保存模型
    autoencoder.save('stackAE.h5')

    # 对测试集进行编码
    encode_images = encoder.predict(x_test)
    plot_representation(encode_images, y_test)

    # 画测试图像
    decode_images = autoencoder.predict(x_test)
    show_images(decode_images, x_test)

