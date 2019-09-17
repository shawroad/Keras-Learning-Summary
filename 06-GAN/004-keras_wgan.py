"""

@file  : 004-keras_wgan.py

@author: xiaolu

@time  : 2019-09-11

"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import RMSprop, SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import math
import keras.backend as K
from keras.utils import plot_model

# wgan的改进: ①: 损失函数使用w值定义; ②:训练d之后 修正参数 wgan的精髓之一


def wasserstein(y_true, y_pred):
    '''
    定义w距离为损失函数
    :param y_true: 真实值
    :param y_pred: 预测值
    :return:
    '''
    return K.mean(y_true * y_pred)


def generator_model():
    '''
    生成模型
    :return:
    '''
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 7 * 7))
    # model.add(BatchNormalization())  # wgan不需要批量归一化
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7, )))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    plot_model(model, show_shapes=True, to_file='./wgan/keras-wgan-generator_model.png')
    return model


def discriminator_model():
    '''
    判别模型
    :return:
    '''
    model = Sequential()

    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
    # model.add(BatchNormalization())  # wgan不需要批量归一化
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same'))
    # model.add(BatchNormalization())  # wgan不需要批量归一化
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same'))
    # model.add(BatchNormalization())  # wgan不需要批量归一化
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(1, (3, 3), padding='same'))
    model.add(GlobalAveragePooling2D())

    plot_model(model, show_shapes=True, to_file='./wgan/keras-wgan-discriminator_model.png')

    return model


def generator_containing_discriminator(g, d):
    '''
    两个模型合并到一块
    :param g: 生成模型
    :param d: 判别模型
    :return:
    '''
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    plot_model(model, show_shapes=True, to_file='./wgan/keras-wgan-gan_model.png')
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]  # 输出是一个向量
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1: 3]

    image = np.zeros((height * shape[0], width * shape[1]), dtype=generated_images.dtype)

    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]: (i+1) * shape[0], j * shape[1]: (j+1) * shape[1]] = img[:, :, 0]
    return image


def train(BATCH_SIZE):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((-1, 28, 28, 1))

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = X_test.reshape((-1, 28, 28, 1))

    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)

    # 定义优化器
    # d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    # g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    d_optim = RMSprop(lr=5E-5)
    g_optim = RMSprop(lr=5E-5)

    # 梯度裁剪的范围
    c_lower = -0.1
    c_upper = 0.1

    # 编译
    # g.compile(loss='binary_crossentropy', optimizer='SGD')
    g.compile(loss='mse', optimizer=g_optim)

    # d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d_on_g.compile(loss=wasserstein, optimizer=g_optim)

    d.trainable = True
    # 判别模型
    d.compile(loss=wasserstein, optimizer=d_optim)

    for epoch in range(100):
        print('Epoch is {}/100'.format(epoch))
        nb_batches = int(X_train.shape[0] / BATCH_SIZE)

        for index in range(nb_batches):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

            image_batch = X_train[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            if index % 100 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                # 调试阶段不生成图片
                Image.fromarray(image.astype(np.uint8)).save('./wgan/'+str(epoch)+"_"+str(index)+".png")

            X = np.concatenate((image_batch, generated_images))
            y = [-1] * BATCH_SIZE + [1] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)

            # 训练d之后 修正参数 wgan的精髓之一 　梯度裁剪
            for l in d.layers:
                weights = l.get_weights()
                weights = [np.clip(w, c_lower, c_upper) for w in weights]
                l.set_weights(weights)

            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [-1] * BATCH_SIZE)  # 将生成的假图片标记为真
            d.trainable = True
            print("epoch %d/100 batch %d d_loss: %f g_loss : %f " % (epoch+1, index, d_loss, g_loss))

            if index % 10 == 9:
                g.save_weights('./wgan/wgan_generator', True)
                d.save_weights('./wgan/wgan_discriminator', True)


if __name__ == '__main__':
    BATCH_SIZE = 100
    train(BATCH_SIZE)
