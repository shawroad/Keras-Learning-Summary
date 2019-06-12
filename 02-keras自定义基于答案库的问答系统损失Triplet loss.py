"""

@file   : 02-keras自定义基于答案库的问答系统损失Triplet loss.py

@author : xiaolu

@time   : 2019-06-12

"""
'''
基于答案库的问答系统   一般会将问题和答案encoder成等长的向量，然后我们计算两个向量直接的相似度，也就是cos值
我们这里设计一个损失函数，就是让正确答案与问题的cos大于 所有错误答案与该问题的cos值。
即   loss = max(0, m + cos(q, A_wrong) - cos(q, A_right))   m是一个大于零的正数。
m + cos(q,A_wrong) − cos(q, A_right)这部分，我们知道目的是拉大正确与错误答案的差距
cos(q,A_right) − cos(q,A_wrong) > m，也就是差距大于m时，由于max的存在，loss就等于0，这时候就自动达到最小值，
就不会优化它了。所以，triplet loss的思想就是：只希望正确比错误答案的差距大一点（并不是越大越好），
超过m就别管它了，集中精力关心那些还没有拉开的样本吧！
我们已经有问题和正确答案，错误答案只要随机挑就行，所以这样训练样本是很容易构造的。
不过Keras中怎么实现triplet loss呢？看上去是一个单输入、双输出的模型，但并不是那
么简单，Keras中的双输出模型，只能给每个输出分别设置一个loss，然后加权求和，但这
里不能简单表示成两项的加权求和。
'''
from keras.layers import Input, Embedding, LSTM, Dense, Lambda
from keras.layers.merge import dot
from keras.models import Model
from keras import backend as K

word_size = 128
nb_features = 10000
nb_classes = 10
encode_size = 64
margin = 0.1


# 词嵌入 + LSTM
embedding = Embedding(nb_features, word_size)
lstm_encoder = LSTM(encode_size)


# 针对每个输入 进行词嵌入
def encode(input):
    return lstm_encoder(embedding(input))


q_input = Input(shape=(None,))
a_right = Input(shape=(None,))
a_wrong = Input(shape=(None,))


q_encoded = encode(q_input)
a_right_encoded = encode(a_right)
a_wrong_encoded = encode(a_wrong)

# 一般的做法是，直接讲问题和答案用同样的方法encode成向量后直接匹配，但我认为这是不合理的，我认为至少经过某个变换。
q_encoded = Dense(encode_size)(q_encoded)    # 将问题进行简单的线性变化


# 计算cos
right_cos = dot([q_encoded, a_right_encoded], -1, normalize=True)
wrong_cos = dot([q_encoded, a_wrong_encoded], -1, normalize=True)

loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]))([wrong_cos, right_cos])

model_train = Model(inputs=[q_input, a_right, a_wrong], outputs=loss)
model_q_encoder = Model(inputs=q_input, outputs=q_encoded)
model_a_encoder = Model(inputs=a_right, outputs=a_right_encoded)


model_train.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)
model_q_encoder.compile(optimizer='adam', loss='mse')
model_a_encoder.compile(optimizer='adam', loss='mse')

model_train.fit([q, a1, a2], y, epochs=10)

# 其中 q, a1, a2分别是问题、正确答案、错误答案的batch, y是任意形状为(len(q), 1)的矩阵

