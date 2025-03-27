# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

"""
建立网络模型结构
"""


class GetFirst(nn.Module):
    def __init__(self):
        super(GetFirst, self).__init__()

    def forward(self, x):
        # 提取 LSTM output 张量，确保后续层（如 nn.Linear）能正确处理序列的每个时间步
        return x[0]


class SentenceMatchNetwork(nn.Module):
    def __init__(self, config):
        super(SentenceMatchNetwork, self).__init__()
        # 可以用bert，参考下面
        # pretrain_model_path = config["pretrain_model_path"]
        # self.bert_encoder = BertModel.from_pretrained(pretrain_model_path)

        # 常规的embedding + layer
        hidden_size = config["hidden_size"]  # 256
        # 20000应为词表大小，这里借用bert的词表，没有用它精确的数字，因为里面有很多无用词，舍弃一部分，不影响效果
        self.embedding = nn.Embedding(20000, hidden_size)
        # 一种多层按顺序执行的写法，具体的层可以换
        # unidirection:batch_size, max_len, hidden_size
        # bidirection:batch_size, max_len, hidden_size * 2

        # bidirectional=True 明确启用双向LSTM的标志
        # 双向 LSTM 层 输入形状：(batch_size, seq_len, hidden_size)
        # 双向LSTM的输出维度为 hidden_size * 2 batch_size seq_len, hidden_size * 2
        # 双向LSTM通过同时处理序列的正向（从左到右）和反向（从右到左）信息，将两个方向的隐藏状态拼接（concatenate）后作为最终输出。其核心特征是输出的特征维度会翻倍
        # LSTM 的输出结构：
        # output: 所有时间步的隐藏状态，形状为 (batch_size, seq_len, hidden_size*2)。
        # h_n: 最后一个时间步的隐藏状态，形状为 (num_layers*2, batch_size, hidden_size)。
        # c_n: 最后一个时间步的细胞状态，形状同 h_n。

        # nn.ReLU() 输入/输出形状：保持 (batch_size, seq_len, hidden_size*2) 引入非线性，增强模型表达能力。

        # nn.Linear输入形状：(batch_size, seq_len, hidden_size*2)。
        # 输出形状：(batch_size, seq_len, hidden_size)。
        # 作用：将双向 LSTM 输出的拼接特征（维度 2*hidden_size）映射回 hidden_size，减少参数量。
        self.encoder = nn.Sequential(nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True),
                                     GetFirst(),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size * 2, hidden_size),  # batch_size, max_len, hidden_size
                                     nn.ReLU(),
                                     )
        # 输出 logits（未归一化的概率)形状为 (batch_size, seq_len, 2)。
        # 目的：将高维特征（hidden_size 维）映射到分类任务的标签空间（此处是二分类，2 个类别）。
        self.classify_layer = nn.Linear(hidden_size, 2)
        self.loss = nn.CrossEntropyLoss()

    # 同时传入两个句子的拼接编码
    # 输出一个相似度预测，不匹配的概率
    def forward(self, input_ids, target=None):
        # x = self.bert_encoder(input_ids)[1]
        # input_ids = batch_size, max_length

        # print('input_ids', input_ids.shape)  # torch.Size([128, 20])
        x = self.embedding(input_ids)  # x:batch_size, max_length, hidden_size torch.Size([128, 20, 256])
        x = self.encoder(x)  #
        # x: batch_size, max_len, hidden_size
        # print('x', x.shape)  # torch.Size([128, 20, 256])
        x = nn.MaxPool1d(x.shape[1])(x.transpose(1, 2)).squeeze()
        # kernel_size=x.shape[1]) => 20 =>1
        # x.transpose(1, 2)  # 形状变为 (128, 256, 1)
        # .squeeze()  # 移除大小为1的维度，形状变为 (128, 256)
        # 将时间步维度（20）放到最后，适配 MaxPool1d 的通道维度要求
        # x: batch_size, hidden_size  (128, 256)
        x = self.classify_layer(x)
        # x: batch_size, 2  (128, 2)
        # 如果有标签，则计算loss
        if target is not None:
            # print('target', target.shape)  target 形状应为 (128, 1)，squeeze() 后变为 (128,)
            return self.loss(x, target.squeeze())
        # 如果无标签，预测相似度
        else:
            # dim=-1 表示在 最后一个维度 上计算 softmax。
            # 对于形状为 (batch_size, 2) 的 x，最后一个维度是 2（即类别维度）
            return torch.softmax(x, dim=-1)[:, 1]  # 如果改为x[:,0]则是两句话不匹配的概率


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SentenceMatchNetwork(Config)
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    l = torch.LongTensor([[1], [0]])
    # y = model(s1, s2, l)
    # print(y)
    # print(model.state_dict())
