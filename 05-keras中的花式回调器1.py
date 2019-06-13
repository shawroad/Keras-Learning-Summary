"""

@file   : 05-keras中的花式回调器1.py

@author : xiaolu

@time   : 2019-06-13

"""
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
x_train = x_train / 255.
x_test = x_test / 255.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(64, kernel_size=3, strides=(1, 1), padding='valid', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=3, strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 根据验证集的指标来保留最优模型，最简便的方法是通过自带的ModelCheckpoint
checkpoint = ModelCheckpoint(filepath="./best_model.weights",
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

model.fit(x_train, y_train, epochs=1, batch_size=64, validation_data=(x_test, y_test), callbacks=[checkpoint])



