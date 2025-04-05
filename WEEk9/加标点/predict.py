# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel
"""
模型效果测试
"""

class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        # print('self.schema', self.schema)  # {'': 0, '，': 1, '。': 2, '？': 3}
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        # print('self.index_to_sign', self.index_to_sign)  # {0: '', 1: '，', 2: '。', 3: '？'}
        self.vocab = self.load_vocab(config["vocab_path"])
        self.model = TorchModel(config)
        # self.model 是当前模型的实例,load_state_dict 是 torch.nn.Module 类的一个方法，用于加载模型的状态字典。状态字典是一个字典，包含了模型的所有参数（如权重和偏置）及其对应的值
        self.model.load_state_dict(torch.load(model_path))

        self.model.eval()
        print("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def predict(self, sentence):
        input_id = []
        for char in sentence:
            # input_id 创建了一个形状为`(1, sequence_length)`的张量
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        # print('input_id', torch.LongTensor([input_id]).shape)  # torch.Size([1, 33]) 或者 torch.Size([1, 23]) shape:(1, sen_len)
        with torch.no_grad():
            y = self.model(torch.LongTensor([input_id])) # 全部
            # print('y.shape', y.shape)  # torch.Size([1, 33, 4])
            # 去除批处理维度：[0] 的作用是从输出中提取第一个（且唯一一个）样本的预测结果，将形状从 (1, 33, 4) 变为 (33, 4)
            res = self.model(torch.LongTensor([input_id]))[0]
            # dim=-1 表示在每一行中寻找最大值的索引
            res = torch.argmax(res, dim=-1)
            # print('res', res)  # tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 2])
        labeled_sentence = ""
        for char, label_index in zip(sentence, res):
            labeled_sentence += char + self.index_to_sign[int(label_index)]
        return labeled_sentence


if __name__ == "__main__":
    sl = SentenceLabel(Config, "model_output/epoch_10.pth")

    sentence = "客厅的颜色比较稳重但不沉重相反很好的表现了欧式的感觉给人高雅的味道"
    res = sl.predict(sentence)
    print(res)  # 客厅的颜色比较稳重，但不沉重相反很好的表现了欧式的感觉给人高雅的味道。

    sentence = "双子座的健康运势也呈上升的趋势但下半月有所回落"
    res = sl.predict(sentence)
    print(res)  # 双子座的健康运势也呈上升的趋势，但下半月有所回落。
