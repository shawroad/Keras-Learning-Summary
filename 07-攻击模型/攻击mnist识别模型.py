"""

@file  : 攻击mnist识别模型.py

@author: xiaolu

@time  : 2019-09-11

"""
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout
from keras.layers.core import Flatten
from keras.datasets import mnist
import keras
import numpy as np
from PIL import Image
import math
import keras.backend as K
from keras.utils.generic_utils import Progbar
import os
from keras.utils import plot_model
from keras.utils import to_categorical


def trainCNN():
    '''
    建立一个卷积模型去识别mnist数据集
    :return:
    '''
    # 1. 定义一些超参数
    batch_size = 128
    num_classes = 10
    epochs = 10

    # 2. 加载数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32').reshape((-1, 28, 28, 1)) / 255.
    x_test = x_test.astype('float32').reshape((-1, 28, 28, 1)) / 255.

    print("x_train shape:", x_train.shape)
    print('train samples:', x_train.shape[0])
    print('test samples:', x_test.shape[0])

    # 将标签转为one_hot
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # 3. 建立模型
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.summary()
    plot_model(model, show_shapes=True, to_file='./hackMnist/keras-cnn.png')

    # 保证只有第一次调用的时候会训练参数
    if os.path.exists('./hackMnist/keras-cnn.h5'):
        model.load_weights('./hackMnist/keras-cnn.h5')
    else:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        model.save_weights("./hackMnist/keras-cnn.h5")

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


def hackAll():
    '''
    攻击训练好的模型
    :return:
    '''
    # 1. 获取非零的100张的图片
    raw_images = getDataFromMnist()

    generator_images = np.copy(raw_images)  # 拷贝一份 对这些图片做梯度改变
    # print(generator_images.shape)  # (100, 28, 28, 1)

    image = get_images(generator_images)

    image = (image / 2.0 + 0.5) * 255.
    Image.fromarray(image.astype(np.uint8)).save('./hackMnist/100mnist-row.png')

    image = 255.0 - image
    Image.fromarray(image.astype(np.uint8)).save('./hackMnist/100mnist-row-w.png')

    cnn = trainCNN()

    # 把那些非零的手写数字都伪装成0
    object_type_to_fake = 0

    model_input_layer = cnn.layers[0].input
    model_output_layer = cnn.layers[-1].output

    cost_function = model_output_layer[0, object_type_to_fake]
    gradient_function = K.gradients(cost_function, model_input_layer)[0]
    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                    [cost_function, gradient_function])
    # learning_rate = 0.3
    e = 0.007

    # progress_bar = Progbar(target=100)
    for index in range(100):
        mnist_image_raw = generator_images[index]
        mnist_image_hacked = np.copy(mnist_image_raw)

        mnist_image_hacked = np.expand_dims(mnist_image_hacked, axis=0)

        preds = cnn.predict(mnist_image_hacked)
        # print("preds:{} mnist:{} likehood:{}\n".format(preds, np.argmax(preds), np.amax(preds)))
        # 调整的极限 彩色图片
        # max_change_above = mnist_image_raw + 0.01
        # max_change_below = mnist_image_raw - 0.01
        # 调整的极限 灰度图片
        max_change_above = mnist_image_raw + 1.0
        max_change_below = mnist_image_raw - 1.0

        i = 0
        cost = 0
        while cost < 0.80:

            cost, gradients = grab_cost_and_gradients_from_model([mnist_image_hacked, 0])
            n = np.sign(gradients)
            mnist_image_hacked += n * e

            mnist_image_hacked = np.clip(mnist_image_hacked, max_change_below, max_change_above)
            mnist_image_hacked = np.clip(mnist_image_hacked, -1.0, 1.0)
            i += 1
            print('step: %d, loss: %f' % (i, cost))

        # 覆盖原有图片
        generator_images[index] = mnist_image_hacked

    # 保存图片
    image = get_images(generator_images)
    image = (image / 2.0 + 0.5) * 255.0
    Image.fromarray(image.astype(np.uint8)).save("./hackMnist/100mnist-hacked.png")

    image = 255.0 - image

    Image.fromarray(image.astype(np.uint8)).save("./hackMnist/100mnist-hacked-w.png")

    # 为了让图片好看　我们进行后处理
    # 灰度图像里面黑是0 白是255 可以把中间状态的处理下
    image[image > 127] = 255
    Image.fromarray(image.astype(np.uint8)).save("./hackMnist/100mnist-hacked-w-good.png")


if __name__ == '__main__':
    # 1. 训练图片识别模型
    trainCNN()
    # 2. 攻击当前模型
    hackAll()

