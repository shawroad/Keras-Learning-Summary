"""

@file   : 2-训练好的词向量添加到Embedding层.py

@author : xiaolu

@time   : 2019-06-11

"""
from gensim.models.word2vec import Word2Vec
import numpy as np
from keras.layers import Embedding

# 1.加载模型
path = './data/model.txt'
word2vecModel = Word2Vec.load(path)

# gensim的word2vec模型 把所有的单词和词向量都存储在了Word2VecModel.wv里面

# # 做个小测试
# vector = word2vecModel.wv['位置']
# print(vector)    # 打印出词向量  和第一次打印的结果相同


# for i, j in word2vecModel.wv.vocab.items():
#     print(i)   # i是词
#     print(j)   # j 代表封装了词频等信息的 gensim“Vocab”对象，例子：Vocab(count:1481, index:38, sample_int:3701260191


# 2.构造“词语-词向量”字典
# 2.1 词列表
vocab_list = [word for word, vocab in word2vecModel.wv.vocab.items()]
word_index = {" ": 0}  # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
word_vector = {}    # 初始化`[word : vector]`字典

# 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
# 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
embeddings_matrix = np.zeros((len(vocab_list) + 1, word2vecModel.vector_size))
# 2.2 填充字典和矩阵
for i in range(len(vocab_list)):
    word = vocab_list[i]   # 每个词语
    word_index[word] = i + 1  # 词语：序号
    word_vector[word] = word2vecModel.wv[word]  # 词语：词向量
    embeddings_matrix[i + 1] = word2vecModel.wv[word]   # 词向量矩阵


# 3. 开始往keras的Embedding中扔
EMBEDDING_DIM = 50  # 词向量维度
MAX_SEQUENCE_LENGTH = 10  # 当前每次输入文本的长度   这里是随便设置的

embedding_layer = Embedding(input_dim=len(embeddings_matrix),   # 字典长度
                            output_dim=EMBEDDING_DIM,   # 词向量 长度（100）
                            weights=[embeddings_matrix],    # 重点：预训练的词向量系数
                            input_length=MAX_SEQUENCE_LENGTH,   # 每句话的 最大长度（必须padding）
                            trainable=False   # 是否在 训练的过程中 更新词向量
                            )


