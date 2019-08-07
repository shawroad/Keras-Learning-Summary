"""

@file   : 13-知识蒸馏.py

@author : xiaolu

@time   : 2019-08-07

"""
from keras.datasets import mnist
from keras.layers import *
from keras.models import Model
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
# 之所以要引入tensorflow主要是想让此代码在GPU上执行
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf



# 定义复杂模型(老师模型)
def teacher_model():
    input_ = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), padding='same')(input_)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    out = Dense(10, activation='softmax')(x)
    model = Model(inputs=input_, outputs=out)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    return model


# 定义简单模型(学生模型)让其学习老师的输出
def student_model():
    input_ = Input(shape=(28, 28, 1))
    x = Flatten()(input_)
    x = Dense(512, activation='sigmoid')(x)
    out = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_, outputs=out)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    model.summary()

    return model


# 老师教的学生
def teach_student(teacher_out, student_model, x_train, x_test, y_test):
    # 1. 获取到老师的输出
    t_out = teacher_out

    # 2. 将学生模型重新训练  所有的参数都进行更新
    s_model = student_model
    for l in s_model.layers:
        l.trainable = True

    # 3. 将标签转为了one_hot
    label_test = to_categorical(y_test)

    model = Model(s_model.input, s_model.output)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam")
    model.fit(x_train, t_out, batch_size=64, epochs=5)

    s_predict = np.argmax(model.predict(x_test), axis=1)
    s_label = np.argmax(label_test, axis=1)
    print(accuracy_score(s_predict, s_label))


if __name__ == '__main__':

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)  # 设置session
    ktf.set_session(sess)
    # 先训练老师模型,如果老师模型一旦训练好, 然后将老师模型的所有输出作为学生模型学习的输出,最后训练出一个跟老师差不多的模型
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.

    t_model = teacher_model()   # 测试集上做到了98%的准确性
    t_model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_test, y_test))

    s_model = student_model()   # 测试机准确率最高也才达到0.9460
    s_model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_test, y_test))

    t_out = t_model.predict(x_train)

    teach_student(t_out, student_model, x_train, x_test, y_test)

