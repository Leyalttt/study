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
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            # 中 B-ORGANIZATION
            # 国 I-ORGANIZATION
            # 政 I-ORGANIZATION
            # 府 I-ORGANIZATION
            # 对 O
            # 目 B-TIME
            # 前 I-TIME
            # 南 B-LOCATION
            # 亚 I-LOCATION
            # 出 O
            # 现 O
            segments = f.read().split("\n\n")  # 来分句子
            # segment 一行
            for segment in segments:
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    # 默认按空格分割字符串，返回一个列表
                    char, label = line.split()
                    sentenece.append(char)
                    # 通过schema答案集将label转成数字
                    # schema {
                    #   "B-LOCATION": 0,
                    #   "B-ORGANIZATION": 1,
                    #   "B-PERSON": 2,
                    #   "B-TIME": 3,
                    #   "I-LOCATION": 4,
                    #   "I-ORGANIZATION": 5,
                    #   "I-PERSON": 6,
                    #   "I-TIME": 7,
                    #   "O": 8
                    # }
                    labels.append(self.schema[label])
                # print('labels0', len(labels)) # 每个句子的label数量
                # 如果模型使用bert 要考虑前后的cls 和sep
                self.sentences.append("".join(sentenece))
                # print('self.sentences', self.sentences) # [',通过演讲赛、法律知识竞赛,举行法律小品晚会等形式,对官兵进行经常性教育,大力加强了官兵的法纪观念,提高了整个部队的法制建设', '....',...]
                input_ids = self.encode_sentence(sentenece)

                labels = self.padding(labels, -1)  # 超过100阶段 不够100 -1补全
                # print('labels0', torch.LongTensor(labels).shape)  # torch.Size([100])
                # print('torch.LongTensor(labels)', torch.LongTensor(labels))
                # tensor([ 0,  4,  8,  8,  1,  5,  5,  8,  8,  8,  8,  2,  6,  6,  3,  7,  7,  8,
                #          1,  5,  5,  5,  5,  5,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  0,
                #          4,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
                #          8,  8,  8,  8,  0,  4,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
                #          8,  8,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                #         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
                # print('torch.LongTensor(input_ids)', torch.LongTensor(input_ids))
                # tensor([ 185,  868, 1289, 4483, 3254,  677,  868,  269, 2626,  269, 3652, 2223,
                #          868, 1802,   19,   21, 1853,  876, 3254,  677,  868, 1117, 2626,  292,
                #          643, 3711, 1663,  489,   13,  868, 4307, 2911,  292, 1315, 1598, 4377,
                #         2282,  868, 1142, 2806, 2142, 2805,  320, 1299, 3027, 2769,  643, 1202,
                #          291,  302,   13,  977,  538, 1797, 1661,  727, 1287,  546, 4377, 2282,
                #          868, 1142, 3158, 1611,  727, 1299,  144, 3792, 2198,  643, 1202, 2769,
                #          547,  538,  145,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                #            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                #            0,    0,    0,    0])
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
            # print('self.data0', len(self.data))  # 408个句子 1412个句子
        return

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            # chars.txt
            for char in text:
                # print('char0', char) # 一个字
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

    def load_schema(self, path):
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
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("../ner_data/train.txt", Config)
