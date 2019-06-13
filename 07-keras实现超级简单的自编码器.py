"""

@file   : 07-keras实现超级简单的自编码器.py

@author : xiaolu

@time   : 2019-06-13

"""
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.models import Model
from keras import backend as K


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)    # (60000, 28, 28)
print(x_test.shape)    # (10000, 28, 28)

x_train = x_train.reshape((60000, 28*28))
x_test = x_test.reshape((10000, 28*28))

x_train = x_train / 255.
x_test = x_test / 255.

# 定义模型
x_in = Input(shape=(784,))
x = x_in

# 编码
x = Dense(256, activation='relu')(x)
x = Dense(64, activation='tanh')(x)
# 解码
x = Dense(256, activation='relu')(x)
x = Dense(784, activation='tanh')(x)

model = Model(x_in, x)
loss = K.mean((x_in - x)**2)

# 将自定义的损失加入模型
model.add_loss(loss)
model.compile(optimizer='adam')

# fit的时候，原来的目标数据，现在是None，因为这种方式已经把所有的输入输出都通过Input传递进来了。
model.fit(x_train, None, epochs=2)   # 因为模型的输入输出都是x_train

# 模型保存
model.save("autoencoder.h5")

encoded_imgs = model.predict(x_test)

