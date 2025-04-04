# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.config["class_num"] = len(self.schema)
        self.max_length = config["max_length"]  # 50
        self.load()

    def load(self):
        self.data = []
        # 训练集: 新浪体育讯北京时间11月8日消息，据《休斯敦纪事报》消息，火箭在昨天客场加时121-124不敌马刺，至此他们开局五连败。.....
        with open(self.path, encoding="utf8") as f:
            for line in f:
                # 一行内的文字个数大于50
                if len(line) > self.max_length:
                    """
                    计算当前行可分割为多少个完整的 max_length 长度的块，并循环处理这些块
                    len(line) // max_length = 10 // 3 = 3 → 循环 3 次，分割为 3 块（每块 3 字符），剩余 1 字符未处理。
                    """
                    for i in range(len(line) // self.max_length):
                        #input_id 文字所对应的idx, label:符号对应的下表, 文字是0
                        input_id, label = self.process_sentence(line[i * self.max_length:(i + 1) * self.max_length])
                        self.data.append([torch.LongTensor(input_id), torch.LongTensor(label)])
                else:
                    # input_id 文字所对应的idx, label:符号对应的下表, 文字是0, 不够的-1补
                    input_id, label = self.process_sentence(line)
                    self.data.append([torch.LongTensor(input_id), torch.LongTensor(label)])
                # print('self.data0', self.data)
        #         [tensor([ 240, 3027,  212, 1152,  697,  290, 3845, 3594, 2766, 3845, 3136, 1289,
        #         4440, 2676, 2143, 3064, 4440, 2552,  212, 1930,  546,  225,  475, 1174,
        #         4582, 2816,  873, 2769, 1372, 1547, 2682,  225, 4552, 3594, 2769, 3594,
        #         2766, 3845, 3128, 1547,  727,  249,  315, 2769, 1928, 1716, 4010,    0,
        #            0,    0]),
        #         tensor([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,
        #          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,
        #          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1, -1])]
        #     print('len(self.data)0', len(self.data))  # 31353和7088
        return

    def process_sentence(self, line):
        sentence_without_sign = []
        label = []  # 标点所对应 self.schema的下标和不是标点的文字0
        # print('line', line)
        # line[:-1]: 截取字符串 line 的从头开始到倒数第二个字符的子串（即去掉最后一个字符)
        for index, char in enumerate(line[:-1]):
            # char 当前字, 如果当前字符是标点（定义在 self.schema 中），跳过处理
            # char 是和 self.schema 的键（keys）进行比较
            if char in self.schema:  # 准备加的标点，在训练数据中不应该存在
                continue
            # char添加到sentence_without_sign中(文字和标点)
            sentence_without_sign.append(char)
            # print('sentence_without_sign', sentence_without_sign)  # ['有', '一', '天', ',', '某', '团', '四']...
            # next_char 下一个字
            next_char = line[index + 1]
            # 下一个字是标点, 将self.schema中所对应的value放入label数组中
            if next_char in self.schema:  # 下一个字符是标点，计入对应label(1或2或3)
                label.append(self.schema[next_char])
            else:
                label.append(0)
        assert len(sentence_without_sign) == len(label)
        # encode_sentence 文字所对应的idx
        encode_sentence = self.encode_sentence(sentence_without_sign)
        label = self.padding(label, -1)
        # print('label0', label)  # [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1....]
        assert len(encode_sentence) == len(label)
        self.sentences.append("".join(sentence_without_sign))  # 所有句子就是整篇文章
        # print('self.sentences0', self.sentences)
        # ['为人谦和,作风民主,在他领导下工作是一种幸福 “文革”时“破四旧”,“上海古旧书店”这个名字就不', '叫了,于是在1967年改名为“上海书店”（1984年又改名为上海图书公司）当时也有红卫兵来店里要'....]
        # encode_sentence 文字所对应的idx, label:符号对应的下表, 文字是0, 不够的-1补
        return encode_sentence, label

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    # 答案集
    def load_schema(self, path):
        # {
        #   "": 0,
        #   "，": 1,
        #   "。": 2,
        #   "？": 3
        # }
        with open(path, encoding="utf8") as f:
            return json.load(f)


# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    # dg 是class DataGenerator的实例
    dg = DataGenerator(data_path, config)
    # DataLoader 数据加载器
    # dg 数据  batch_size 一个batch中的数据数量(128)  shuffle = True 随机取数
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    # 表示 一个 epoch 中需要迭代的批次数，而非样本总数, __len__函数中的31353和7088 / 128 向上取整
    # print('len(dl)', len(dl))  # 245和56
    # 遍历dl验证al有没有被DataLoader加载
    # for each in dl:
    #     print('each', each)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("../ner_data/train.txt", Config)
