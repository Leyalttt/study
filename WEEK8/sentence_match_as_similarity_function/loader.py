# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import logging
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import BertTokenizer

"""
数据加载
train.json 训练数据
valid.json 测试数据
"""

logging.getLogger("transformers").setLevel(logging.ERROR)


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.tokenizer.vocab)  # 21128
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"]  # 由于采取随机采样，所以需要设定一个采样数量，否则可以一直采 10000
        self.max_length = config["max_length"]  # 20
        self.data_type = None  # 用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                # 因为文件是json格式
                line = json.loads(line)
                # 训练数据和测试数据的每一行
                # print('line', line)
                # 加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    # 多个question
                    questions = line["questions"]
                    # print('questions', questions)  # 取出当前行问题
                    label = line["target"]
                    # print('label', label)  # 取出当前行目标 比如:宽泛业务问题
                    for question in questions:
                        # schema=> {'停机保号': 0, '密码重置': 1, '宽泛业务问题': 2, '亲情号码设置与修改': 3, '固话密码修改': 4,
                        # self.schema[label] = schema中的value 就是 索引,
                        # value(索引)作为key, 每个问题作为value
                        self.knwb[self.schema[label]].append(question)
                    # print('self.knwb', self.knwb)  # defaultdict(<class 'list'>, {2: ['问一下我的电话号', '办理业务', '办理相关业务',....], 14: ['改下畅聊套餐'],...}
                # 加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    # print('line', line) # ["其他业务", "宽泛业务问题"]
                    question, label = line
                    # self.schema[label] label作为key, 找到对应的value(是数字)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([question, label_index])
                    print('self.data', self.data)
                    # print('len(self.data)', len(self.data))  # 464
        #             [['其他业务', tensor([2])], ['手机信息', tensor([2])], ['语音查话费', tensor([23])],
        return

    # 每次加载两个文本，输出他们的拼接后编码
    def encode_sentence(self, text1, text2):
        # truncation='longest_first' 表示如果编码后的输入 ID 长度超过了 max_length，则从最长的序列开始截断。padding='max_length' 表示如果编码后的输入
        # ID 长度小于 max_length，则在末尾用 0 填充 将 text1 和 text2 拼接在一起，并在它们之间添加了 [SEP] 分隔符，然后将拼接后的文本编码为输入
        # ID拼接格式：[CLS] + s1 + [SEP] + s2 + [SEP]。
        input_id = self.tokenizer.encode(text1, text2,
                                         truncation='longest_first',
                                         max_length=self.max_length,
                                         padding='max_length',
                                         )
        return input_id

    # 确定数据集长度, 从而计算每个epoch的迭代次数
    def __len__(self):
        if self.data_type == "train":
            # 10000
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            # 返回实际加载的样本数
            # print('len(self.data)', len(self.data))
            return len(self.data)
    # 用于从数据集中获取单个样本或批次数据, 动态生成数据
    # 每个批次（batch）会调用 batch_size 次 __getitem__（例如 batch_size=128 时，79 个批次对应 79×128=10112 次调用，但实际受 epoch_data_size 限制为 10000 次）
    def __getitem__(self, index):
        if self.data_type == "train":
            # 上面长度是10000, 实际数据没有10000条, 采用random_train_sample 通过随机采样动态生成训练样本：
            # 优势：即使原始数据量少，也能生成大量训练样本，避免模型过拟合
            return self.random_train_sample()  # 随机生成一个训练样本
        else:
            return self.data[index]

    # 依照一定概率生成负样本或正样本
    # 负样本从随机两个不同的标准问题中各随机选取一个
    # 正样本从随机一个标准问题中随机选取两个
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        # self.knwb中的key
        # print('standard_question_index', standard_question_index)  # [2, 23, 22, 15, 12, 17, 9, 20, 10, 27, 26, 6, 3, 19, 25, 4, 5, 28, 11, 0, 18, 13, 8, 7, 16, 24, 14, 21, 1]
        # 随机正样本(句子相似)
        # positive_sample_rate => 0.5
        # random.random() 生成了一个在 [0.0, 1.0) 范围内的随机浮点数
        if random.random() <= self.config["positive_sample_rate"]:
            # 随机选取standard_question_index不放回
            p = random.choice(standard_question_index)
            # 如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
            if len(self.knwb[p]) < 2:
                return self.random_train_sample()
            else:
                # 从 standard_question_index 列表中随机选择了两个元素(就是两个问题) 并将这两个元素分别赋值给 s1 和 s2 不放回
                s1, s2 = random.sample(self.knwb[p], 2)
                # s1, s2 拼接在一起同时传给模型, s1, s2 会遇见
                input_ids = self.encode_sentence(s1, s2)
                # print('input_ids', input_ids)
                # [101, 2769, 7444, 6206, 3341, 4510, 3227, 4850, 1216, 5543, 102, 5543, 679, 5543, 1215, 3341, 4510, 6413, 749, 102]
                input_ids = torch.LongTensor(input_ids)
                # 值为 1，表示这是一个正样本。

                # 例如:采样句子 ["如何办理停机保号", "停机保号流程"]，编码后标签为 1。
                return [input_ids, torch.LongTensor([1])]
        # 随机负样本(句子不相似)
        else:
            # 随机选取standard_question_index中的两个不放回
            p, n = random.sample(standard_question_index, 2)
            s1 = random.choice(self.knwb[p])
            s2 = random.choice(self.knwb[n])
            input_ids = self.encode_sentence(s1, s2)
            input_ids = torch.LongTensor(input_ids)
            # 值为 0，表示这是一个负样本。
            # tensor([ 101, 2506,  928,  102, 4510,  928,  689, 1218,  102,    0,    0,    0,
            #            0,    0,    0,    0,    0,    0,    0,    0])
            # print('input_ids', input_ids)

            # 例如 采样句子 ["停机保号流程", "密码重置步骤"]，编码后标签为 0。
            return [input_ids, torch.LongTensor([0])]


# 加载字表或词表
# 在使用 BERT 模型的分词器
def load_vocab(vocab_path):
    tokenizer = BertTokenizer(vocab_path)
    # print('tokenizer', tokenizer)
    # BertTokenizer(name_or_path='', vocab_size=21128, model_max_length=1000000000000000019884624838656, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=True, added_tokens_decoder={
    # 	0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    # 	100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    # 	101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    # 	102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    # 	103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    # }
    # )
    # print('len(tokenizer).vocab', len(tokenizer.vocab))  # 21128
    return tokenizer


# 加载schema 相当于答案集
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        # print('json.loads(f.read())', json.loads(f.read()))
        # 停机保号就是train中的target
        # {'停机保号': 0, '密码重置': 1, '宽泛业务问题': 2, '亲情号码设置与修改': 3, '固话密码修改': 4, '来电显示开通': 5, '亲情号码查询': 6, '密码修改': 7, ..
        return json.loads(f.read())


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    # dg 是class DataGenerator的实例
    dg = DataGenerator(data_path, config)
    # DataLoader 数据加载器
    # dg 数据  batch_size 一个batch中的数据数量(128)  shuffle = True 随机取数
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    # 表示 一个 epoch 中需要迭代的批次数，而非样本总数, __len__函数中的10000 / 128 向上取整就是79
    # print('len(dl)', len(dl))  # 79
    # 遍历dl验证al有没有被DataLoader加载
    # for each in dl:
    #     print('each', each)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("../data/valid.json", Config)
