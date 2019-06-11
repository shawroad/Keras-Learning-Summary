"""

@file   : 3-加载卷积层部分再自定义全连接层.py

@author : xiaolu

@time   : 2019-06-11

"""
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, Model
from keras.datasets import mnist
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = load_model('mnist_LeNet5.h5')
model.summary()
print("\n"*3)


# 把全连接层全部扔掉
base_model = Model(inputs=model.input, outputs=model.get_layer('max_pooling2d_2').output)
base_model.summary()
print("\n"*3)


final_model = Sequential()
final_model.add(base_model)
final_model.add(Flatten())
final_model.add(Dense(128, activation='relu', trainable=False))
final_model.add(Dropout(0.2, trainable=False))
final_model.add(Dense(10, activation='softmax', trainable=False))
final_model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=64, validation_data=(x_test, y_test))

# 保存模型
model.save("improve_model.h5")

