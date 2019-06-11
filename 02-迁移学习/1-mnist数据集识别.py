"""

@file   : 1-mnist数据集识别.py

@author : xiaolu

@time1  : 2019-06-10

"""
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense


# 加载数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)    # (60000, 28, 28)
print(x_test.shape)    # (10000, 28, 28)


# 对数据进行预处理
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
x_train = x_train.astype('float32') / 255
x_test = x_test.astype("float32") / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 建立模型
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(10, activation='softmax'))    # mnist数据集有是个类别 所以最后Dense输出10

# 打印一下模型结构
model.summary()

# 编译模型 并训练
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=64, validation_data=(x_test, y_test))

# 保存模型
model.save('mnist_LeNet5.h5')