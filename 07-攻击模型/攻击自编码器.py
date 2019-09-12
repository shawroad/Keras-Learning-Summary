"""

@file  : 攻击自编码器.py

@author: xiaolu

@time  : 2019-09-12

"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers.core import Flatten
from keras.optimizers import SGD,RMSprop,adam
from keras.losses import binary_crossentropy
from keras.datasets import mnist
import keras
import numpy as np
from PIL import Image
import argparse
import math
import keras.backend as K
from keras.utils.generic_utils import Progbar
import os
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import plot_model


def trainAutoEncodeNosiy():
    '''
    类似于一个去噪自编码器
    :return:
    '''
    # 1. 定义一些超参数
    batch_size = 128
    epochs = 20

    # 2. 加载数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 28 * 28))
    x_test = x_test.reshape((-1, 28 * 28))

    # 3. 对图像做简单预处理
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train -= 0.5
    x_train *= 2.0
    x_test -= 0.5
    x_test *= 2.0

    # 4. 对原始图片加入噪声
    x_train_nosiy = x_train + 0.1 * np.random.normal(loc=0., scale=1., size=x_train.shape)
    x_test_nosiy = x_test + 0.1 * np.random.normal(loc=0., scale=1., size=x_test.shape)
    x_train_nosiy = np.clip(x_train_nosiy, -1., 1.)
    x_test_nosiy = np.clip(x_test_nosiy, -1., 1.)

    print(x_train.shape)
    print(x_train_nosiy.shape)

    # 5. 定义模型
    input_img = Input(shape=(28*28, ))
    encoded = Dense(100, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)

    model = Model(inputs=[input_img], outputs=[decoded])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()

    plot_model(model, show_shapes=True, to_file='./hackAutoEncode/keras-ae.png')

    # 保证只有第一次调用的时候会训练参数
    h5file = "./hackAutoEncode/autoEncodeNosiy.h5"
    if os.path.exists(h5file):
        model.load_weights(h5file)
    else:
        model.fit(x_train_nosiy, x_train, epochs=epochs,
                  batch_size=batch_size, verbose=1,
                  validation_data=(x_test_nosiy, x_test))
        model.save_weights(h5file)
    return model


def trainAutoEncode():
    batch_size = 128
    epochs = 20

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 28 * 28))
    x_test = x_test.reshape((-1, 28 * 28))
    input_shape = (28*28, )

    # 图像转换到-1到1之间
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train -= 0.5
    x_train *= 2.0
    x_test -= 0.5
    x_test *= 2.0

    input_img = Input(shape=input_shape)
    encoded = Dense(100, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)
    model = Model(inputs=[input_img], outputs=[decoded])
    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.summary()
    plot_model(model, show_shapes=True, to_file='./hackAutoEncode/keras-ae.png')

    # 保证只有第一次调用的时候会训练参数
    h5file = "./hackAutoEncode/autoEncode.h5"
    if os.path.exists(h5file):
        model.load_weights(h5file)
    else:
        model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
                  verbose=1, validation_data=(x_test, x_test))

        model.save_weights(h5file)
    return model


def getDataFromMnist():
    # 获取100个非0样本
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 原有范围在0-255转换到 0-1
    # x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # 原有范围在0-255转换调整到-1和1之间
    x_train = x_train.astype(np.float32) / 255.0
    x_train -= 0.5
    x_train *= 2.0

    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    index = np.where(y_train != 0)  # 非零为True 是零则为False
    x_train = x_train[index]
    x_train = x_train[-100:]  # 获取最后面的一百张图片

    return x_train


def get_images(generated_images):
    # 将获取的100张图片画到一张图上
    num = generated_images.shape[0]  # 100 是我们选中的那一百张图片
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1: 3]
    image = np.zeros((height * shape[0], width * shape[1]), dtype=generated_images.dtype)

    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]: (i+1) * shape[0], j * shape[1]: (j+1) * shape[1]] = img[:, :, 0]

    return image


def getZeroFromMnist():
    '''
    获取数字0的图案
    :return:
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_train -= 0.5
    x_train *= 2.0

    index = np.where(y_train == 0)
    x_train = x_train[index]
    x_train = x_train[-1:]

    return x_train


def hackAutoEncode():
    raw_images = getDataFromMnist()
    generator_images = np.copy(raw_images)
    # image = get_images(generator_images)
    # image = (image / 2.0 + 0.5) * 255.0
    # Image.fromarray(image.astype(np.uint8)).save('./hackAutoEncode/100mnist-raw.png')
    # image = 255.0 - image
    # Image.fromarray(image.astype(np.uint8)).save('./hackAutoEncode/100mnist-raw-w.png')

    # 将获取的图片reshape成想要的形状
    generator_images = generator_images.reshape((100, 784))

    # 获取模型
    model = trainAutoEncode()

    # 都伪装成0
    object_type_to_fake = getZeroFromMnist()  # 获取标签为0的图片
    object_type_to_fake = object_type_to_fake.reshape((28 * 28, ))
    object_type_to_fake = np.expand_dims(object_type_to_fake, axis=0)

    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output

    # 生成的图像与图案0之间的差为损失函数 下面的三种损失函数都可以用
    # cost_function = binary_crossentropy(y_pred,object_type_to_fake)
    cost_function = K.mean(K.square(model_output_layer-object_type_to_fake))
    # cost_function = K.mean(K.binary_crossentropy(model_output_layer, object_type_to_fake))

    gradient_function = K.gradients(cost_function, model_input_layer)[0]
    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                    [cost_function, gradient_function])
    cost = 0.0
    e = 0.007
    for index in range(100):
        mnist_image_raw = generator_images[index]
        mnist_image_hacked = np.copy(mnist_image_raw)
        mnist_image_hacked = mnist_image_hacked.reshape(28*28, )
        mnist_image_hacked = np.expand_dims(mnist_image_hacked, axis=0)

        # 调整的极限 灰度图片
        max_change_above = mnist_image_raw + 1.0
        max_change_below = mnist_image_raw - 1.0

        i = 0
        cost = 100
        while cost > -12.0 and i < 500:
            cost, gradients = grab_cost_and_gradients_from_model([mnist_image_hacked, 0])

            n = np.sign(gradients)
            mnist_image_hacked -= n * e

            mnist_image_hacked = np.clip(mnist_image_hacked, max_change_below, max_change_above)
            mnist_image_hacked = np.clip(mnist_image_hacked, -1.0, 1.0)

            print("image_seg: {}, batch:{} Cost: {:.8}%".format(index, i, cost * 100))
            i += 1

        # 覆盖原有图片
        generator_images[index] = mnist_image_hacked

    autoEncode_images = np.copy(generator_images)
    generator_images = generator_images.reshape(100, 28, 28, 1)

    # 保存图片
    image = get_images(generator_images)
    image = (image/2.0+0.5)*255.0
    Image.fromarray(image.astype(np.uint8)).save("./hackAutoEncode/100mnist-hacked.png")

    # image=255.0-image
    # Image.fromarray(image.astype(np.uint8)).save("hackAutoEncode/100mnist-hacked-w.png")

    # 灰度图像里面黑是0 白是255 可以把中间状态的处理下
    # image[image>127]=255
    # Image.fromarray(image.astype(np.uint8)).save("hackAutoEncode/100mnist-hacked-w-good.png")

    for index in range(100):
        mnist_image_raw = autoEncode_images[index]
        mnist_image_hacked = np.copy(mnist_image_raw)
        mnist_image_hacked=mnist_image_hacked.reshape(28*28)
        mnist_image_hacked = np.expand_dims(mnist_image_hacked, axis=0)
        preds = model.predict(mnist_image_hacked)

        autoEncode_images[index] = preds

    autoEncode_images = autoEncode_images.reshape((100, 28, 28, 1))
    image = get_images(autoEncode_images)
    image = (image / 2.0 + 0.5) * 255.0
    Image.fromarray(image.astype(np.uint8)).save("./hackAutoEncode/100mnist-hacked-autoEncode-b.png")

    image = 255.0-image
    Image.fromarray(image.astype(np.uint8)).save("./hackAutoEncode/100mnist-hacked-autoEncode-w.png")

    # 灰度图像里面黑是0 白是255 可以把中间状态的处理下
    image[image > 127] = 255
    Image.fromarray(image.astype(np.uint8)).save("./hackAutoEncode/100mnist-hacked-autoEncode-w-good.png")


if __name__ == "__main__":
    # trainCNN()
    # hackAll()
    hackAutoEncode()


