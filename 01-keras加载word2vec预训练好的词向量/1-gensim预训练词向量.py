"""

@file   : 1-gensim预训练词向量.py

@author : xiaolu

@time   : 2019-06-11

"""
import jieba
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence

def load_data(path):
    with open(path, 'r', encoding='gbk') as f:
        text = f.read()
        text = text.strip().replace('\n', '')
    return text


def split_word(text):
    # 使用结巴进行分词
    word_list = jieba.lcut(text)
    temp = ' '.join(word_list)

    # 写入文件
    temp = temp.encode('utf8')
    with open('./data/corpus_split.txt', 'wb') as f:
        f.write(temp)


def train_vec():
    model = Word2Vec(LineSentence('./data/corpus_split.txt'), hs=1, min_count=1, window=3, size=50)

    model_file_name = './data/model.txt'
    model.save(model_file_name)    # 保存训练好的模型

    # model = Word2Vec.load(model_file_name)  # 加载训练好的模型


def load_model_train():
    # 加载模型。 并看一个词的词向量
    test_word = "位置"
    model = Word2Vec.load("./data/model.txt")
    print("看看“位置”这个词的词向量:", model[test_word])
    print("与“位置”这个词最近的5个词:", model.most_similar(test_word, topn=5))


if __name__ == '__main__':
    # path = './data/corpus.txt'
    # text = load_data(path)
    #
    # # 开始分词
    # split_word(text)
    #
    # # 训练词向量
    # train_vec()

    # 测试词向量
    load_model_train()
