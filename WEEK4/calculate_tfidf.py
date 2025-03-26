import jieba
import math
import os
import json
from collections import defaultdict

"""
tfidf的计算和使用
"""

#统计tf和idf值
def build_tf_idf_dict(corpus):
    # print('corpus', corpus)  # [['世界', '，', '尽', '在于', '心', '：', '全新', '梅赛德斯', '-', '奔驰', 'S', '级', '轿车', '
    # print('len', len(corpus))  # 6
    tf_dict = defaultdict(dict)  #key:文档序号，value：dict，文档中每个词出现的频率   (字典里包着字典)
    idf_dict = defaultdict(set)  #key:词， value：set，文档序号，最终用于计算每个词在多少篇文档中出现过  集合可以自动去重  (字典里包着集合)
    # 遍历分词, 长度为6的数组
    for text_index, text_words in enumerate(corpus):
        # word 单个分词
        for word in text_words:
            # 计算某类别(text_index)某词次数
            if word not in tf_dict[text_index]:
                tf_dict[text_index][word] = 0
            tf_dict[text_index][word] += 1
            # 将当前文本的索引添加到 idf_dict 字典中对应词的集合中(如果该次在当前文档里)
            # print('idf_dict[word]', idf_dict[word])  # {0, 1, 2, 3, 4}....
            idf_dict[word].add(text_index)
    # print('idf_dict', idf_dict) # {'世界': {0, 3, 5}, '，': {0, 1, 2, 3, 4, 5}, '尽': {0}, '在于': {0},
    # 转成该次包含文档的个数
    idf_dict = dict([(key, len(value)) for key, value in idf_dict.items()])
    # print('idf_dict', idf_dict)  # {'世界': 3, '，': 6, '尽': 1, '在于': 1, '心': 1, '：':

    return tf_dict, idf_dict

#根据tf值和idf值计算tfidf
def calculate_tf_idf(tf_dict, idf_dict):
    tf_idf_dict = defaultdict(dict)
    # print('tf_dict', tf_dict)  # {0: {'世界': 2, '，': 4, '尽': 1, '在于': 1, '心': 1
    for text_index, word_tf_count_dict in tf_dict.items():
        for word, tf_count in word_tf_count_dict.items():
            # word_tf_count_dict 字典value的和
            tf = tf_count / sum(word_tf_count_dict.values())
            #tf-idf = tf * log(D/(idf + 1))
            tf_idf_dict[text_index][word] = tf * math.log(len(tf_dict)/(idf_dict[word]+1))
    return tf_idf_dict

#输入语料 list of string
#["xxxxxxxxx", "xxxxxxxxxxxxxxxx", "xxxxxxxx"]
def calculate_tfidf(corpus):
    #先进行分词
    corpus = [jieba.lcut(text) for text in corpus]
    # print('corpus', corpus)  # 分词 [['世界', '，', '尽', '在于', '心', '：', '全新', '梅赛德斯', '-', '奔驰', 'S'
    tf_dict, idf_dict = build_tf_idf_dict(corpus)
    tf_idf_dict = calculate_tf_idf(tf_dict, idf_dict)
    return tf_idf_dict

#根据tfidf字典，显示每个领域topK的关键词
def tf_idf_topk(tfidf_dict, paths=[], top=10, print_word=True):
    # print('tfidf_dict', tfidf_dict)  # {0: {'世界': 0.0007819963512211464, '，': -0.0005946024294204758, '尽': 0.0010594139717146672,
    topk_dict = {}
    for text_index, text_tfidf_dict in tfidf_dict.items():
        #
        word_list = sorted(text_tfidf_dict.items(), key=lambda x:x[1], reverse=True)
        topk_dict[text_index] = word_list[:top]
        if print_word:
            # print(text_index, paths[text_index])
            # for i in range(top):
                # print(word_list[i])
            #     ('特斯拉', 0.008689405349353219)
            # ('汽车', 0.005864972634158598)
            # ('车', 0.00534732636883275)
            # ('进军', 0.005297069858573335)
            # ('新能源', 0.005297069858573335)
            # ('万辆', 0.005297069858573335)
            # ('奔驰', 0.004237655886858669)
            # ('现代', 0.004237655886858669)
            # ('自动', 0.004237655886858669)
            # ('驾驶', 0.004237655886858669)
            # ----------
            print("----------")
    return topk_dict

def main():
    dir_path = r"category_corpus/category_corpus"
    corpus = []
    paths = []
    # 列出目录 dir_path 下的所有文件和子目录列表返回
    # print('os.listdir(dir_path)', os.listdir(dir_path))  # ['auto.txt', 'finance.txt', 'health.txt', 'science.txt', 'sports.txt', 'world.txt']
    for path in os.listdir(dir_path):
        # 将目录路径和文件名拼接成完整的文件路径
        path = os.path.join(dir_path, path)
        if path.endswith("txt"):
            corpus.append(open(path, encoding="utf8").read())
            paths.append(os.path.basename(path))
    # print('corpus', corpus) # 文本内容['世界，尽在于心：全新梅赛德斯-奔驰S级轿车今日上市\n下一个特斯....
    # print('len', len(corpus)) # 6, 6个文本文件, append6次
    tf_idf_dict = calculate_tfidf(corpus)
    tf_idf_topk(tf_idf_dict, paths)

if __name__ == "__main__":
    main()
