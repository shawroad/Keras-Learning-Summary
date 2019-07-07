"""

@file   : 12-双管齐下LSTM+CNN进行文本分类.py

@author : xiaolu

@time   : 2019-07-07

"""
from keras.datasets import reuters
from keras.layers import *
import numpy as np
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import Model
from keras.utils.vis_utils import plot_model


# 我们用CNN和LSTM 分别进行学习 最后融合 然后加Dense
def build_model(maxlen):
    # 先定义CNN
    input = Input(shape=(maxlen, ))
    embedding = Embedding(input_dim=10000, output_dim=128)
    e = embedding(input)
    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(e)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D(name='CNN_output')(x)

    # 定义BiLSTM
    y = Bidirectional(LSTM(64, activation='tanh', return_sequences=True))(e)
    y = Bidirectional(LSTM(32, activation='tanh', return_sequences=False, name='LSTM_output'))(y)

    # 将两种方式的输出按0.5, 0.5的权重相加
    x = Lambda(lambda x: 0.5*x)(x)
    y = Lambda(lambda y: 0.5*y)(y)
    total = Add(name="model_merge")([x, y])

    dense1 = Dense(64, activation='tanh')(total)
    output = Dense(46, activation='softmax')(dense1)

    model = Model(inputs=input, outputs=output)
    plot_model(model=model, to_file='test.png', show_shapes=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, batch_size=64, validation_data=(x_test, y_test))

    # 保存模型
    model.save('reuters.h5')


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)
    print(x_train.shape)  # (8982,)
    print(x_test.shape)  # (2246,)
    print(np.unique(y_train))  # 0 ——> 45 总共46种类别

    maxlen = 300
    x_train = sequence.pad_sequences(x_train, padding='post', maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, padding='post', maxlen=maxlen)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    build_model(maxlen)


