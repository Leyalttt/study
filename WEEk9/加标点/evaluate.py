# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        # valid_data_path->【导读】哪些方法可以预防小儿脑瘫?小儿脑瘫是一种非常常见的小儿疾病，相信很多人都不希望自己的孩子得这种病
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.schema = self.valid_data.dataset.schema
        self.index_to_label = dict((y, x) for x, y in self.schema.items())
        # print('self.index_to_label0', self.index_to_label)  # {0: '', 1: '，', 2: '。', 3: '？'}

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = dict(zip(self.schema.keys(), [defaultdict(int) for i in range(len(self.schema))]))
        # print('self.stats_dict0', self.stats_dict)  # {'': defaultdict(<class 'int'>, {}), '，': defaultdict(<class 'int'>, {}), '。': defaultdict(<class 'int'>, {}), '？': defaultdict(<class 'int'>, {})}
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            # 取出部分句子  batch_size:128
            sentences = self.valid_data.dataset.sentences[
                        index * self.config["batch_size"]: (index + 1) * self.config["batch_size"]]
            # print('sentences0', sentences)  # ['【导读】哪些方法可以预防小儿脑瘫?小儿脑瘫是一种非常常见的小儿疾病相信很多人都不希望自己的孩子得',..]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            # print('input_id0', len(input_id))  # 128
            with torch.no_grad():
                pred_results = self.model(input_id)  # 不输入labels，使用模型当前参数进行预测
                # print('pred_results', pred_results)
                #     tensor([[[ 1.7303e+00, -8.1674e-01, -3.6548e-01, -2.8251e+00],
                #          [ 1.4543e+00, -3.0260e-01, -2.8353e-01, -3.3795e+00],
                #          [ 3.0755e+00, -1.0377e+00, -1.5318e+00, -4.6773e+00],
                #          ...,
                # print('pred_results,shape', pred_results.shape)  # torch.Size([128, 50, 4])
                # print('len(pred_results)', len(pred_results)) # 128
                # print('sentences', len(sentences))  # 128
            self.write_stats(labels, pred_results, sentences)
        self.show_stats()
        return

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences), print(len(labels), len(pred_results), len(sentences))
        if not self.config["use_crf"]:
            # dim=-1 表示在每一行中寻找最大值的索引
            pred_results = torch.argmax(pred_results, dim=-1)
            # print('pred_results', pred_results)
            #     tensor([[0, 0, 0,  ..., 0, 0, 0],
            #         [0, 0, 0,  ..., 0, 0, 0],
            #         [0, 0, 0,  ..., 0, 0, 0],
            #         ...,
            #         [0, 0, 0,  ..., 0, 0, 0],
            #         [0, 0, 0,  ..., 0, 0, 0],
            #         [0, 0, 0,  ..., 0, 0, 0]])
            # print('pred_results.shape', pred_results.shape)  # torch.Size([128, 50])
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                # [:len(sentence)] 确保列表的长度与句子长度相同
                pred_label = pred_label.cpu().detach().tolist()[:len(sentence)]
            true_label = true_label.cpu().detach().tolist()[:len(sentence)]
            for pred, gold in zip(pred_label, true_label):
                # key:对应的符号
                key = self.index_to_label[gold]
                # print('key', key) # , 或者   。  或者  ? 或者空
                self.stats_dict[key]["correct"] += 1 if pred == gold else 0
                self.stats_dict[key]["total"] += 1
        return

    def show_stats(self):
        total = []
        for key in self.schema:
            acc = self.stats_dict[key]["correct"] / (1e-5 + self.stats_dict[key]["total"])
            self.logger.info("符号%s预测准确率：%f" % (key, acc))
            total.append(acc)
        self.logger.info("平均acc：%f" % np.mean(total))
        self.logger.info("--------------------")
        return
