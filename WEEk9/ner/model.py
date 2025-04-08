# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF

"""
CRF有很多种第三方库, pip install pytorch-crf, 已经封装好了维特比, 使用时self.crf_layer._viterbi_decode(predict)
建立网络模型结构
序列标注训练
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]  # 9个分类
        num_layers = config["num_layers"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        # 与文本分类的区别, 没有pooling, 取lstm最后一个字的向量代表整句话的向量做概率分布, 文本分类一句话分一次, 序列标注一句话10个字做10次分类
        # hidden_size * 2 因为LSTM用的是双向
        self.classify = nn.Linear(hidden_size * 2, class_num)
        # class_num: 类别数量（即标签的总数，如 B-PER, I-PER, O 等）
        # batch_first=True: 输入数据的维度顺序为 (ses)batch_size, sequence_length, num_clas
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]  # 用或者不用crf
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        x = self.embedding(x)  # input shape:(batch_size, sen_len)
        x, _ = self.layer(x)  # input shape:(batch_size, sen_len, input_dim)
        predict = self.classify(x)  # ouput:(batch_size, sen_len, num_tags)
        # print('predict', predict.shape)  # torch.Size([16, 100, 9]) batch_size:16 batch_size, sequence_length, num_clas
        # 对比使用和不使用crf的效果
        if target is not None:
            if self.use_crf:
                # 前面label用-1补全
                # 生成一个与 target 形状相同的布尔张量（mask），其中：
                # mask(掩码) 中对应位置为 True：表示 target 中该位置的值 大于 -1。
                # mask 中对应位置为 False：表示 target 中该位置的值 小于或等于 -1
                # 形状: (batch_size, sequence_length)
                # 含义:用于区分有效字符和填充字符（Padding）。
                #
                # 大于-1表示有效位置，小于等于-1表示填充位置。
                # target => tensor([ 0,  4,  8,  8,  1,  5,  5,  8,  8,  8,  8,  2,  6,  6,  3,  7,  7,  8,
                #          1,  5,  5,  5,  5,  5,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  0,
                #          4,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
                #          8,  8,  8,  8,  0,  4,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
                #          8,  8,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                #         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
                mask = target.gt(-1)
                # print('mask', mask)
                # tensor([[ True,  True,  True,  ..., False, False, False],
                #         [ True,  True,  True,  ..., False, False, False],
                #         [ True,  True,  True,  ..., False, False, False],
                #         ...,
                #         [ True,  True,  True,  ..., False, False, False],
                #         [ True,  True,  True,  ..., False, False, False],
                #         [ True,  True,  True,  ..., False, False, False]])

                # 加个负号 因为在这个crf用的相反数作为loss, 其他库可能不需要这个负号
                # predict(模型输出):形状为 (batch_size, seq_len, num_tags)。
                # target(真实标签):形状为 (batch_size, seq_len)。

                # reduction="mean": 对批次内所有样本的损失取平均
                # print('target0', target.shape)  # torch.Size([16, 100])  batch_size, sen_len
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                # (number, class_num), (number)
                # 通过调整输入张量的形状，使其适配标准的交叉熵损失函数
                # 输入要求：
                # predict（预测值）：形状 (N, C)，其中 N 是样本数，C 是类别数。
                # target（标签）：形状 (N,)，每个值是类别索引（0 ≤ 值 < C）。

                # 输出：标量损失值，表示所有样本的平均损失。
                # predict 从 (batch_size, seq_len, num_tags) 展平为 (batch_size*seq_len, num_tags)，
                # target.view(-1)展平为一维张量  target 从 (batch_size, seq_len) 展平为 (batch_size*seq_len,)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    model = TorchModel(Config)
