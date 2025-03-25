#!/usr/bin/env python3  
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

# 输入模型文件路径
# 加载训练好的模型 word2vec 是一个用于训练词向是模型的工具 而不是一个预训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)  # 加载模型
    # print('model', model)  # Word2Vec<vocab=19322, vector_size=128, alpha=0.025>
    return model

# 加载所有句子进行句子分词
def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            # strip 去除空格
            sentence = line.strip()
            # 分词用空格连接
            sentences.add(" ".join(jieba.cut(sentence)))
    # print('sentences', sentences)  # {'高血压 药膳 食疗   帮 你 降低 血压', '姚明 ： 火箭 目前为止 不够格   希望 背靠背 两场 都 能 打'.....} 不同的句子用逗号隔开
    # len() 函数来获取集合（set）的长度 不重复
    # print("句子数量：", len(sentences))  # 1796个句子
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    # print('sentences', len(sentences))   # 1796个句子的分词  {'高血压 药膳 食疗   帮 你 降低 血压', '姚明 ： 火箭 目前为止 不够格   希望 背靠背 两场 都 能 打'.....}
    # 遍历set类型 1796个句子的分词
    for sentence in sentences:
        # print('sentence', sentence)  # 一个句子的分词, 如: ['高血压', '药膳', '食疗', '帮', '你', '降低', '血压']
        words = sentence.split()  # sentence是分好词的，空格分开
        # numpy 库的 zeros 函数创建一个全为零的数组，数组的长度为 model.vector_size
        # model.vector_size 将返回模型中每个词向量的维度
        vector = np.zeros(model.vector_size)  # 128
        # 所有词的向量相加求平均，作为句子向量
        # print('words', words)  # ['高血压', '药膳', '食疗', '帮', '你', '降低', '血压']
        for word in words:
            # words一句话中的所有分词, word 一个分词
            try:
                # 是 Gensim 库中 Word2Vec 模型的一个常用方法，用于获取给定单词的词向量表示
                # model.wv[word] 会返回word的的词向量,vector 的前 128 个元素将是 model.wv[word] 的值
                vector += model.wv[word]
                # print('vector', vector)  #[] 128个数
                # print('len', len(vector))  128
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        # 将 vector所有词的词向量 除以一句话的分词数量，得到该句子的平均词向量
        vectors.append(vector / len(words))
        # print('len(words)', len(words)) 一句话中有几个分词
    # 将 vectors 列表转换为 NumPy 数组并返回
    # print('vectors', vectors)  # 1796个句子的词向量 [5.42226471e-03,  1.27053463e-01,  5.89003444e-02, -8.72810513e-02,...]
    # print('lenvectors', len(vectors))  # 1796个句子
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"D:\TTT\NLP算法\预习\week5 词向量及文本向量\model.w2v")  # 加载训练好的词向量模型
    # titles.txt
    # 新增资金入场 沪胶强势创年内新高
    # NYMEX原油期货电子盘持稳在75美元下方
    # 2010艺术品秋拍上演六宗最
    # 4月粗钢产量创新高 业内人士预计钢价将扬升
    sentences = load_sentence("titles.txt")  # 加载所有标题进行分词
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    # print("指定聚类数量：", n_clusters)  # 42
    # 创建一个 K-Means 聚类器对象，其中 n_clusters 是你希望聚类的簇（或类）的数量
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    # 使用 fit 方法对词向量进行聚类
    kmeans.fit(vectors)  # 进行聚类计算

    # 创建一个默认字典，用于存储每个簇的句子
    sentence_label_dict = defaultdict(list)
    # sentences => {'高血压 药膳 食疗   帮 你 降低 血压', '姚明 ： 火箭 目前为止 不够格   希望 背靠背 两场 都 能 打'.....} 不同的句子用逗号隔开
    # 对应的簇标签列表kmeans.labels_, 是一个整数数组，其中的每个元素都是一个整数，表示对应样本所属的簇的标签
    # 是一个长度为 1796 的数组，每个元素表示对应句子被分配到的簇标签（范围是 0 到 41，共 42 个簇）
    # print('kmeans.labels_', kmeans.labels_)  # [35 26 23 ... 17 32 11]
    # print('len(kmeans.labels_)', len(kmeans.labels_))  # 1796
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
    # print('sentence_label_dict', sentence_label_dict)  # defaultdict(<class 'list'>, {12: ['一种 抚慰 人心 的 安静 （ 图 ）', '个性化 的 窗帘   选择 一块 设计 （ 图 ）'
    # print('len', len(sentence_label_dict)) # 42
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)  # cluster 37 : 分在了37簇
        # 遍历当前簇中的前 10 个句子（如果句子数量少于 10 个，则打印所有句子）。
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            # 打印当前句子的内容，并移除所有空格
            print(sentences[i].replace(" ", ""))  # 高血压药膳食疗帮你降低血压
        print("---------")
if __name__ == "__main__":
    main()
