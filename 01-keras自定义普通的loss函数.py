"""

@file   : 01-keras自定义普通的loss函数.py

@author : xiaolu

@time   : 2019-06-12

"""
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras import backend as K


word_size = 128   # 词嵌入的维度
nb_features = 10000    # 词表的大小
nb_classes = 10   # 类别数
encode_size = 64   # LSTM的输出维度

input = Input(shape=(None, ))
embedded = Embedding(nb_features, word_size)(input)
encoder = LSTM(encode_size)(embedded)
predict = Dense(nb_classes, activation='softmax')(encoder)


# 自定义损失函数
# 因为交叉损失计算是有缺陷的。 如果一旦一个概率大，它就尽可能的让这个概率更大，更接近1 这样损失就最小了
# 我们可以分一点力气，让他雨露均沾
def mycrossentropy(y_true, y_pred, e=0.1):
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / nb_classes, y_pred)
    return (1-e) * loss1 + e * loss2


model = Model(inputs=input, outputs=predict)
model.compile(optimizer='adam', loss=mycrossentropy)









