"""

@file   : 03-keras自定义Dense层.py

@author : xiaolu

@time   : 2019-06-13

"""
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.layers import Layer
from keras import backend as K
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)    # (60000, 28, 28)
print(x_test.shape)    # (10000, 28, 28)
print(y_train.shape)
print(y_test.shape)

x_train = x_train.reshape((60000, 28*28))
x_test = x_test.reshape((10000, 28*28))

x_train = x_train / 255.
x_test = x_test / 255.


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 自己定义一个类似Dense的层
class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        super(MyLayer, self).__init__(**kwargs)   # 必须
        self.output_dim = output_dim   # 可以自定义一些属性，方便调用 这里的作用是指定这一层的输出维度

    def build(self, input_shape):
        # 添加可训练的参数
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),   # shape[1]指定是输入的列数
                                      initializer='uniform',
                                      trainable=True
                                      )

    def call(self, x, **kwargs):
        # 定义功能, 这一层相当于Lambda层的功能函数
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        # 计算输出形状，如果输入和输出形状一致，那么可以省略，否则最好加上
        return (input_shape[0], self.output_dim)


model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(28*28,)))
model.add(Dense(64, activation='relu'))
model.add(MyLayer(output_dim=32))    # 用自己自定义层
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1, batch_size=64, validation_data=(x_test, y_test))

model.save('mnist-recognize.h5')
