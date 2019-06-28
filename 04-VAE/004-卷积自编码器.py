"""

@file   : 004-卷积自编码器.py

@author : xiaolu

@time   : 2019-06-28

"""
from keras.layers import *
from keras.models import Model
from keras.datasets import mnist


def conv_ae(x_train):

    input_image = Input(shape=(28, 28, 1))

    # encoding layer
    x = Conv2D(16, (3, 3), activation='relu', padding="same")(input_image)
    x = MaxPool2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPool2D((2, 2), padding='same')(x)

    # decoding layer
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs=input_image, outputs=decoded)
    encoder = Model(inputs=input_image, outputs=encoded)

    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    autoencoder.summary()

    autoencoder.fit(x_train, x_train, epochs=1, batch_size=64, shuffle=True, )

    return autoencoder, encoder


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # 训练
    encoder, autoencoder = conv_ae(x_train)

    # 保存模型
    autoencoder.save("conv_ae.h5")





