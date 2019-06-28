"""

@file   : 005-自编码器聚类.py

@author : xiaolu

@time   : 2019-06-28

"""
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist


batch_size = 100
latent_dim = 20
epochs = 50
num_classes = 10
img_dim = 28
filters = 16
intermediate_dim = 256


# 加载数据集  并进行预处理
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((-1, img_dim, img_dim, 1))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((-1, img_dim, img_dim, 1))


# 编码部分
x = Input(shape=(img_dim, img_dim, 1))
h = x
for i in range(2):
    filters *= 2
    h = Conv2D(filters=filters, kernel_size=3, strides=2, padding='same')(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(h)
    h = LeakyReLU(0.2)(h)

h_shape = K.int_shape(h)[1:]   # 为了解码的输出考虑
h = Flatten()(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

encoder = Model(x, z_mean)   # 通常认为z_mean就是隐编码  编码器


# 解码部分
z = Input(shape=(latent_dim,))
h = z
h = Dense(np.prod(h_shape))(h)  # np.prod()不指定维度，则是将所有的数进行相乘
h = Reshape(h_shape)(h)

for i in range(2):
    h = Conv2DTranspose(filters=filters, kernel_size=3, strides=1, padding='same')(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same')(h)
    h = LeakyReLU(0.2)(h)
    filters //= 2

x_recon = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same')(h)

decoder = Model(z, x_recon)  # 解码器
generator = decoder


# 对隐层变量进行分类  相当于降维到10类
z = Input(shape=(latent_dim, ))
y = Dense(intermediate_dim, activation='relu')(z)
y = Dense(num_classes, activation='softmax')(y)
classfier = Model(z, y)     # 隐变量分类器


# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
x_recon = decoder(z)   # 对采样的结果解码
y = classfier(z)   # 对采样的结果进行分类


# 每个类别初始化一组均值
class Gaussian(Layer):
    def __init__(self, num_classes, **kwargs):
        super(Gaussian, self).__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, input_shape):
        latent_dim = input_shape[-1]  # 取出隐层的输出
        self.mean = self.add_weight(name='mean',
                                    shape=(self.num_classes, latent_dim),
                                    initializer='zeros')

    def call(self, inputs, **kwargs):
        z = inputs   # z.shape=(batch_size, latent_dim)
        z = K.expand_dims(z, 1)   # 扩充一维 加上标签
        return z - K.expand_dims(self.mean, 0)   # 这里第0进行扩充 相当于搞批量只是为了计算

    def compute_output_shape(self, input_shape):
        return (None, self.num_classes, input_shape[-1])


gaussian = Gaussian(num_classes)
z_prior_mean = gaussian(z)   # 将隐层的两输入gaussian


# 建立模型
vae = Model(x, [x_recon, z_prior_mean, y])


# 定义损失
z_mean = K.expand_dims(z_mean, 1)
z_log_var = K.expand_dims(z_log_var, 1)

lamb = 2.5    # 这是重构误差的权重，它的相反数就是重构方差，越大意味着方差越小。
xent_loss = 0.5 * K.mean((x - x_recon)**2, 0)  # 重构损失
kl_loss = - 0.5 * (z_log_var - K.square(z_prior_mean))  # 前半部分是隐层拟合的均值 - 后半部分相当于我们根据真实值和拟合的均值算的方差

# K.batch_dot()智能乘法  前半部分[0.73, 0.01, 0.02...十维] [距离类别1的损失， 距离类别2的损失，...十维]
kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1), kl_loss), 0)
cat_loss = K.mean(y * K.log(y + K.epsilon()), 0)
vae_loss = lamb * K.sum(xent_loss) + K.sum(kl_loss) + K.sum(cat_loss)


vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

vae.fit(x_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))

# 保存模型
encoder.save('encoder.h5')
classfier.save('classfier.h5')
vae.save('vae.h5')


means = K.eval(gaussian.mean)   # 算出每个类别的中心点
print("每个类别的中心点:", means)

x_train_encoded = encoder.predict(x_train)
y_train_pred = classfier.predict(x_train_encoded).argmax(axis=1)
x_test_encoded = encoder.predict(x_test)
y_test_pred = classfier.predict(x_test_encoded).argmax(axis=1)

