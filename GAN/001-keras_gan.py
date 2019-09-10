"""

@file  : 001-keras_gan.py

@author: xiaolu

@time  : 2019-09-09

"""
from keras.layers import Activation
# from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.optimizers import SGD
import matplotlib.mlab as MLA


# 均值和方差
mu, sigma = (0, 1)


def x_sample(size=200, batch_size=32):
    '''
    真实样本满足正态分布 平均值维0 方差为1 样本维度200
    :param size:
    :param batch_size:
    :return:
    '''
    x = []
    for _ in range(batch_size):
        x.append(np.random.normal(mu, sigma, size))
    return np.array(x)


def z_sample(size=200, batch_size=32):
    '''
    噪声样本是服从均匀分布的
    :param size:
    :param batch_size:
    :return:
    '''
    z = []
    for _ in range(batch_size):
        z.append(np.random.uniform(-1, 1, size))
    return np.array(z)


def generator_model():
    '''
    生成模型
    :return:
    '''
    model = Sequential()
    model.add(Dense(input_dim=200, units=256))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('sigmoid'))
    # plot_model(model, show_shapes=True, to_file='./keras-gan-generator-model.png')
    return model


def discriminator_model():
    '''
    判别模型
    :return:
    '''
    model = Sequential()
    model.add(Reshape((200, ), input_shape=(200, )))
    model.add(Dense(units=256))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # plot_model(model, show_shapes=True, to_file='./keras-gan-discriminator_model.png')
    return model


def generator_containing_discriminator(g, d):
    '''
    对抗模型
    :param g: 生成模型
    :param d: 判别模型
    :return:
    '''
    model = Sequential()
    model.add(g)
    d.trainable = False  # 判别模型的参数不训练
    model.add(d)
    # plot_model(model, show_shapes=True, to_file='./keras-gan-gan_model.png')
    return model


def show_image(s):
    count, bins, ignored = plt.hist(s, 5, normed=True)
    plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.show()


def save_image(s, filename):
    count, bins, ignored = plt.hist(s, bins=20, normed=True,facecolor='w',edgecolor='b')
    y = MLA.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'g--', linewidth=2)
    # plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.savefig(filename)


def show_init():
    x=x_sample(batch_size=1)[0]
    save_image(x, "./x-0.png")
    z=z_sample(batch_size=1)[0]
    save_image(z, "./z-0.png")


def save_loss(d_loss_list, g_loss_list):
    plt.subplot(2, 1, 1)  # 面板设置成2行1列，并取第一个（顺时针编号）
    plt.plot(d_loss_list, 'yo-')  # 画图，染色
    # plt.title('A tale of 2 subplots')
    plt.ylabel('d_loss')

    plt.subplot(2, 1, 2)  # 面板设置成2行1列，并取第二个（顺时针编号）
    plt.plot(g_loss_list, 'r.-')  # 画图，染色
    # plt.xlabel('time (s)')
    plt.ylabel('g_loss')
    plt.savefig("./loss.png")
    plt.show()


if __name__ == '__main__':
    show_init()

    d_loss_list = []
    g_loss_list = []

    batch_size = 128
    d = discriminator_model()
    g = generator_model()

    d_on_g = generator_containing_discriminator(g, d)

    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    for epoch in range(500):

        noise = z_sample(batch_size=batch_size)   # 生成噪声样本
        image_batch = x_sample(batch_size=batch_size)  # 真实样本
        generated_images = g.predict(noise, verbose=0)  # 在生成模型中生成伪造的图片
        x = np.concatenate((image_batch, generated_images))  # 将伪造的图片与真实图片放在一块
        y = [1]*batch_size+[0]*batch_size
        d_loss = d.train_on_batch(x, y)    # 把判别模型训练好
        # print("d_loss : %f" % d_loss)

        noise = z_sample(batch_size=batch_size)
        d.trainable = False
        g_loss = d_on_g.train_on_batch(noise, [1]*batch_size)  # 锁定判别模型的参数 然后对生成模型训练
        d.trainable = True
        # print("g_loss : %f" % g_loss)
        d_loss_list.append(d_loss)
        g_loss_list.append(g_loss)
        print("Epoch is", epoch, 'd_loss', d_loss, 'g_loss', g_loss)

        if epoch % 100 == 1:
            # 测试阶段
            noise = z_sample(batch_size=1)
            generated_images = g.predict(noise, verbose=0)
            # print generated_images
            save_image(generated_images[0], "./z-{}.png".format(epoch))

    save_loss(d_loss_list, g_loss_list)
