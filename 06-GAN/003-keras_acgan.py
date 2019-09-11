"""

@file  : 003-keras_acgan.py

@author: xiaolu

@time  : 2019-09-09

"""
from PIL import Image
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout, add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np

np.random.seed(0)


def build_generator(latent_size):
    '''
    生成模型
    :param latent_size:
    :return:
    '''
    cnn = Sequential()
    cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(128 * 7 * 7, activation='relu'))
    cnn.add(Reshape((7, 7, 128)))

    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(256, (5, 5), padding="same", kernel_initializer="glorot_normal", activation="relu"))

    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(128, (5, 5), padding="same", kernel_initializer="glorot_normal", activation="relu"))
    cnn.add(Conv2D(1, (2, 2), padding="same", kernel_initializer="glorot_normal", activation="tanh"))

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')
    # 对类别进行词嵌入 然后和隐层信息进行对应位相加
    cls = Flatten()(Embedding(10, 100, embeddings_initializer="glorot_normal")(image_class))
    h = add([latent, cls])
    fake_image = cnn(h)

    model = Model(inputs=[latent, image_class], outputs=[fake_image])
    model.summary()

    return Model(inputs=[latent, image_class], outputs=[fake_image])


def build_discriminator():
    '''
    生成判别模型
    :return:
    '''
    cnn = Sequential()

    cnn.add(Conv2D(32, (3, 3), padding="same", strides=(2, 2), input_shape=(28, 28, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, (3, 3), padding="same", strides=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256, (3, 3), padding="same", strides=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(28, 28, 1))

    features = cnn(image)

    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(10, activation='softmax', name='auxiliary')(features)

    return Model(inputs=[image], outputs=[fake, aux])


if __name__ == '__main__':
    # 一:定义一些超参数
    nb_epochs = 1   # 训练epoch
    batch_size = 32  # batch_size的大小
    latent_size = 100   # 隐层的维度
    # 这里是定义优化器的参数
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # 二:建立几个模型
    # 1. 判别模型
    discriminator = build_discriminator()
    discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])

    # 2. 生成模型
    generator = build_generator(latent_size)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss='binary_crossentropy')

    # 3. 组装模型
    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1, ), dtype='int32')
    fake = generator([latent, image_class])  # 组装模型的前半部分
    discriminator.trainable = False  # 组装模型的后半部分是判别模型　并且判别模型的参数要锁定不能让其训练
    fake, aux = discriminator(fake)  # 输出为真假和类别
    combined = Model(inputs=[latent, image_class], outputs=[fake, aux])  # 将生成模型和判别模型拼接到一块
    combined.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])

    # 三: 加载数据
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((-1, 28, 28, 1))
    # X_train = np.expand_dims(X_train, axis=1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = X_test.reshape((-1, 28, 28, 1))
    # X_test = np.expand_dims(X_test, axis=1)
    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    # train_history = defaultdict(list)
    # test_history = defaultdict(list)
    #
    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(X_train.shape[0] / batch_size)  # 算一下大概有多少batch

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))  # 生成一个批量噪音

            image_batch = X_train[index * batch_size: (index + 1) * batch_size]  # 真实图片
            label_batch = y_train[index * batch_size: (index + 1) * batch_size]  # 对应的标签

            sampled_labels = np.random.randint(0, 10, batch_size)  # 随机采样一个batch大小的标签
            generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))])

            # 将生成的fake image 和　real image 合并
            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * batch_size + [0] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            # 开始训练判别模型
            d_loss = discriminator.train_on_batch(X, [y, aux_y])
            epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))  # 输出是一个列表[总损失, 损失1, 损失2] 每步的输出都是这样

            # 组合模型的判别部分的参数锁定　然后生成噪声　并将其标记为真　开始训练组合模型
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, 10, 2 * batch_size)
            trick = np.ones(2 * batch_size)
            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))
            g_loss = combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels])
            print('epoch: {}, step: {}, 判别模型损失:{}, 生成模型损失:{}'.format(epoch, index, d_loss[0], g_loss[0]))

        # 看测试集
        print('\nTesting for epoch {}:'.format(epoch + 1))
        # 噪声 + 随机采样的标签
        noise = np.random.uniform(-1, 1, (nb_test, latent_size))
        sampled_labels = np.random.randint(0, 10, nb_test)
        generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=False)
        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)
        discriminator_test_loss = discriminator.evaluate(X, [y, aux_y], verbose=False)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # 保存权重　一个epoch保存一次模型
        generator.save_weights('params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights('params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        # generate some digits to display
        noise = np.random.uniform(-1, 1, (100, latent_size))
        sampled_labels = np.array([[i] * 10 for i in range(10)]).reshape(-1, 1)  # 标签0到9各生成十个
        generated_images = generator.predict([noise, sampled_labels], verbose=0)

        img = (np.concatenate([r.reshape(-1, 28) for r in np.split(generated_images, 10)], axis=-1) * 127.5 + 127.5).astype(np.uint8)
        Image.fromarray(img).save('plot_epoch_{0:03d}_generated.png'.format(epoch))
