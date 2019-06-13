"""

@file   : 04-自定义层并将center-loss加到层中.py

@author : xiaolu

@time   : 2019-06-13

"""
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.models import Sequential, Model
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

y_test.astype('int32')
y_train.astype('int32')


class Dense_with_Center_loss(Layer):
    def __init__(self, output_dim, **kwargs):
        super(Dense_with_Center_loss, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        # 添加可训练的参数
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim, ),
                                    initializer='zeros',
                                    trainable=True)

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer='glorot_normal',
                                       trainable=True)

    def call(self, inputs, **kwargs):
        # 对于center loss来说，返回结果还是跟Dense的返回结果一致
        # 所以还是普通的矩阵乘法加上偏置
        self.inputs = inputs
        return K.dot(inputs, self.kernel) + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def loss(self, y_true, y_pred, lamb=0.5):
        # 定义完整的loss
        y_true = K.cast(y_true, 'int32')   # 保证y_true的dtype为int32  cast将张量转换为不同的类型返回
        crossentropy = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        centers = K.gather(self.centers, y_true[:, 0])  # 取出样本中心   gather(vector_m, index)在给定的张量中搜索给定下标的向量
        center_loss = K.sum(K.square(centers - self.inputs), axis=1)  # 计算center loss
        return crossentropy + lamb * center_loss


f_size = 20
x_in = Input(shape=(784, ))
f = Dense(f_size)(x_in)

dense_center = Dense_with_Center_loss(10)
output = dense_center(f)

model = Model(x_in, output)
model.compile(loss=dense_center.loss, optimizer='adam', metrics=['sparse_categorical_accuracy'])


# 这里的y_train是类别的整数id, 不用转为one_hot
model.fit(x_train, y_train, epochs=1, batch_size=64, validation_data=(x_test, y_test))

model.save('mnist-recognize.h5')
