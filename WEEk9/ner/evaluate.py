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
        # 训练数据
        # 法 O
        # 律 O.....
        # 邓 B-PERSON
        # 小 I-PERSON
        # 平 I-PERSON
        # 同 O.....
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        # tensor([ 185,  868, 1289, 4483, 3254,  677,  868,  269, 2626,  269, 3652, 2223,
        #          868, 1802,   19,   21, 1853,  876, 3254,  677,  868, 1117, 2626,  292,
        #          643, 3711, 1663,  489,   13,  868, 4307, 2911,  292, 1315, 1598, 4377,
        #         2282,  868, 1142, 2806, 2142, 2805,  320, 1299, 3027, 2769,  643, 1202,
        #          291,  302,   13,  977,  538, 1797, 1661,  727, 1287,  546, 4377, 2282,
        #          868, 1142, 3158, 1611,  727, 1299,  144, 3792, 2198,  643, 1202, 2769,
        #          547,  538,  145,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        #            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        #            0,    0,    0,    0])

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            # print('batch_data', batch_data)
            # 字的下标还有label
            # [tensor([[1129, 3919, 3745,  ...,    0,    0,    0],
            #         [1012,  257,   13,  ...,    0,    0,    0],
            #         [  21,   27, 1853,  ...,    0,    0,    0],
            #         ...,
            #         [ 370,  165, 1880,  ...,    0,    0,    0],
            #         [1018, 1057,  249,  ...,    0,    0,    0],
            #         [  19, 1929,  185,  ...,    0,    0,    0]]), tensor([[ 8,  8,  8,  ..., -1, -1, -1],
            #         [ 3,  7,  8,  ..., -1, -1, -1],
            #         [ 8,  8,  0,  ..., -1, -1, -1],
            #         ...,
            #         [ 8,  8,  8,  ..., -1, -1, -1],
            #         [ 8,  8,  8,  ..., -1, -1, -1],
            #         [ 3,  7,  7,  ..., -1, -1, -1]])]
            """
            1.DataGenerator类继承自Dataset(通过继承 Dataset，DataGenerator 可以直接传递给 DataLoader)：
            DataGenerator是自定义的Dataset子类，它的__getitem__方法会返回预处理后的数据（编码后的input_ids和labels）。同时这个类在self.sentences属性中存储
            了原始句子列表。

            2.DataLoader的创建：
            当通过DataLoader(dg, ...)创建DataLoader对象时，dg这个DataGenerator实例会被保存在DataLoader.dataset属性中。

            3.访问原始数据：
            因此，当你有self.valid_data（这是一个DataLoader对象）时，可以通过self.valid_data.dataset直接访问原始的DataGenerator实例。而DataGenerator中保存了所有原始句子
            的self.sentences列表，所以可以通过self.valid_data.dataset.sentences获取全部句子。
            """
            sentences = self.valid_data.dataset.sentences[
                        index * self.config["batch_size"]: (index + 1) * self.config["batch_size"]]
            # print('sentences', sentences)  # 截取句子 [',通过演讲赛、法律知识竞赛,举行法律小品晚会等形式,对官兵进行经常性教育,大力加强了官兵的法纪观念,提高了整个部队的法制建设,杜绝了各类违法乱纪现象的发生。'....
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            # print('input_id', input_id.shape)  # torch.Size([16, 100])  [batch_size, sentence_len]
            with torch.no_grad():
                pred_results = self.model(input_id)  # 不输入labels，使用模型当前参数进行预测
                # torch.LongTensor([pred_results])，其中外层的 [] 添加了一个多余的维度 1
                # print('pred_results', torch.LongTensor([pred_results]).shape)  # torch.Size([1, 16, 100])
                # print('pred_results', torch.LongTensor(pred_results).shape)  # torch.Size([16, 100])
            self.write_stats(labels, pred_results, sentences)
        self.show_stats()
        return

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            # print('true_label', true_label, 'pred_label', pred_label)
            # true_label [3, 7, 7, 7, 7, 7, 8, 8, 8, 0, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 2, 6, 6, 8, 2, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            # pred_label [3, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            # print("=+++++++++")
            # print('true_entities', true_entities)  # defaultdict(<class 'list'>, {'LOCATION': ['中国'], 'PERSON': ['邓小平']})
            # print('pred_entities', pred_entities)  # defaultdict(<class 'list'>, {})
            # print('=+++++++++')
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                # 预测实体 pred_entities[key] 中，有多少实体也存在于真实实体 true_entities[key] 中。这些实体被认为是正确识别的
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            # 调和平均数，公式为 2 * (P * R) / (P + R)
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            # 宏观平均F1（Macro-F1）
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        # 微观平均F1（Micro-F1）
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum(
            [self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    '''
    {
      "B-LOCATION": 0,     # 地点
      "B-ORGANIZATION": 1,  # 机构
      "B-PERSON": 2,       # 人名
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''

    """
    一个0后面跟着很多4 是一个 LOCATION
    一个1后面跟着很多5 是一个 ORGANIZATION
    """

    # 解码
    def decode(self, sentence, labels):
        # print('labels0', labels)  # [3, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, ....]
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        # print('labels0', labels)  # 377780488888888888888888888
        results = defaultdict(list)
        # 正则模式 04+ 匹配以 0 开头后跟多个 4 的标签序列（如 "0444"）。
        # 匹配到的位置 (s, e) 对应句子中实体的起始和结束索引。
        # re.finditer("(04+)", labels) 每次迭代得到一个 Match 对象，包含匹配的起始位置（start()）和结束位置（end()）
        for location in re.finditer("(04+)", labels):
            # print('location', location)  # <re.Match object; span=(89, 92), match='044'>
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        # print('results', results)  #defaultdict(<class 'list'>, {'LOCATION': ['中国', '非洲', '非洲'], 'ORGANIZATION': ['联合国', '联合国安理会'], 'PERSON': ['沈国放'], 'TIME': ['24日']})
        # print('results', results)  # defaultdict(<class 'list'>, {})
        return results
