# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        # 标签 '差评' 对应的索引是 0，标签 '好评' 对应的索引是 1
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        # 分几类
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        # vocab_path 词表路径
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            print()
            for line in f:
                # 当前行是否以 "0," 开头。如果是，则将 label 设置为 0, line一句话
                # 1 好评 0 差评
                if line.startswith("0,"):
                    label = 0
                elif line.startswith("1,"):
                    label = 1
                else:
                    continue
                # line[2:] 是从第三个字符开始到行末的所有字符，strip() 方法会删除 title 两端的空白字符
                title = line[2:].strip()
                # print(title) 每句话
                if self.config["model_type"] == "bert":
                    # add_special_tokens=True 表示添加特殊的标记，例如 [CLS] 和 [SEP]。max_length=30 表示最大长度为 30。
                    # pad_to_max_length=True 表示如果文本长度小于 30，则用零填充。return_tensors='pt' 表示返回 PyTorch 的张量
                    # tokenizer.encode 方法的返回值是一个列表，包含了输入 ID 和注意力掩码。输入 ID 是一个整数列表，每个整数代表一个单词或子词的 ID。
                    # 注意力掩码是一个布尔列表，用于指示哪些位置是填充的，哪些位置是有效的
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                # 一句话转成id, 不够config里面设置的最大长度30用0补充
                # print('input_id', input_id)  # [101, 2769, 6375, 1914, 1217, 3723, 738, 3766, 1217, 8024, 2582, 720, 1391, 1557, 8043, 2397, 749, 1416, 1543, 4638, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                input_id = torch.LongTensor(input_id)
                # print('input_id---', input_id)  # 30个
                # tensor([ 101, 2769, 6375, 1914, 1217, 3723,  738, 3766, 1217, 8024, 2582,  720,
                #         1391, 1557, 8043, 2397,  749, 1416, 1543, 4638,  102,    0,    0,    0,
                #            0,    0,    0,    0,    0,    0])
                label_index = torch.LongTensor([label])
                # print('label_index', label_index)  # tensor([0])
                #                    x            y
                self.data.append([input_id, label_index])

            # print(len(self.data))  # 训练9589个句子, 测试2398个句子
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        # 这行代码将零添加到 input_id 列表的末尾，直到 input_id 的长度等于 self.config["max_length"]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        # 这行代码遍历文件中的每一行
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    #  {
    #     'apple': 1,
    #     'banana': 2,
    #     'cherry': 3
    # }
    return token_dict


#用torch自带的DataLoader类封装数据
# data_path先是训练数据的地址
def load_data(data_path, config, shuffle=True):
    # 这行代码创建了一个 DataGenerator 对象，用于生成数据。data_path 是数据的路径，config 是一个配置对象，包含了生成数据所需的参数
    # dg是调用DataGenerator类生成的对象, 他有self.data：存储了处理后的数据, self.vocab：自定义词表, self.index_to_label 和 self.label_to_index：标签与索引之间的双向映射等等
    dg = DataGenerator(data_path, config)
    # print('dg', dg)  # <loader.DataGenerator object at 0x0000019D57E1B380>
    # 这行代码创建了一个 DataLoader 对象，用于将数据分批加载到模型中。dg 是数据生成器，batch_size=config["batch_size"] 指定了每个批次的大小，shuffle=shuffle 指定了是否在每个 epoch 开始时打乱数据
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    # print('config["batch_size"]', config["batch_size"])  # 12
    # print('dl', dl) # dl <torch.utils.data.dataloader.DataLoader object at 0x000001FD7F64A2A0>
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_tag_news.json", Config)
    # print('dg--', dg)
    # print(dg[1])
