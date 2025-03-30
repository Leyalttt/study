# -*- coding: utf-8 -*-
import torch
from loader import load_data
import numpy as np

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        # 由于效果测试需要训练集当做知识库，再次加载训练集。
        # 事实上可以通过传参把前面加载的训练集传进来更合理，但是为了主流程代码改动量小，在这里重新加载一遍
        self.train_data = load_data(config["train_data_path"], config)
        self.tokenizer = self.train_data.dataset.tokenizer
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

    # 将知识库中的问题向量化，为匹配做准备
    # 每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮测试都重新进行向量化

    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.questions = []

        # self.train_data.dataset获取的是 DataLoader 使用的自定义 Dataset 实例（假设为 DataGenerator 类）
        # knwb => defaultdict(<class 'list'>, {2: ['问一下我的电话号', '办理业务', '办理相关业务',....], 14: ['改下畅聊套餐'],...}
        # standard_question_index =>schema 答案集target对应的索引
        for standard_question_index, questions in self.train_data.dataset.knwb.items():
            for question in questions:
                # 记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.questions)] = standard_question_index
                self.questions.append(question)
                # print('question_index_to_standard_question_index', self.question_index_to_standard_question_index)
                # {
                #     0: 2,  # 问题0属于标准问题2
                #     1: 2,  # 问题1属于标准问题2
                #     2: 2,  # 问题2属于标准问题2
                #     3: 14  # 问题3属于标准问题14
                # }
                # print('questions', questions)
                # [
                #     '问一下我的电话号',   # 索引 0
                #     '办理业务',          # 索引 1
                #     '办理相关业务',      # 索引 2
                # ]
        # print('self.questions', len(self.questions))  # 1878
        return

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空前一轮的测试结果
        self.model.eval()
        self.knwb_to_vector()
        # print('self.valid_data', self.valid_data) #    <torch.utils.data.dataloader.DataLoader object at 0x0000021F3A15EDB0> [['其他业务', tensor([2])], ['手机信息', tensor([2])], ['语音查话费', tensor([23])],
        for index, batch_data in enumerate(self.valid_data):
            """"
            当使用 DataLoader 加载数据时，假设 batch_size=2，DataLoader 会将两个样本合并为：
            ("其他业务", "手机信息"),          # 字符串合并为元组
            tensor([[2], [2]])               # 标签堆叠为张量
            """
            # print('batch_data', batch_data)
            # [('其他业务', '手机信息', '语音查话费', '报一下我的手机费', '语音播报我的话费是多少'), tensor([[ 2],
            #         [ 2],
            #         [23],
            #         [23],.....
            # 一个batch中有多个问题, 多个label
            test_questions, labels = batch_data
            predicts = []
            for test_question in test_questions:
                input_ids = []
                # 遍历知识库(train)
                for question in self.questions:
                    # 每次加载两个文本，输出他们的拼接后编码, 判断两个句子的相似性, test_question是测试, question是知识库
                    input_ids.append(self.train_data.dataset.encode_sentence(test_question, question))
                # 对 test_question="如何查询话费"，依次与知识库中的每个 question 生成句子对：
                # ["如何查询话费", "查询话费余额"] → 输入编码 input_ids_1。
                # ["如何查询话费", "话费余额查询"] → 输入编码 input_ids_2。
                # ["如何查询话费", "怎么查话费"] → 输入编码 input_ids_3。
                # ["如何查询话费", "停机保号流程"] → 输入编码 input_ids_4。

                with torch.no_grad():
                    input_ids = torch.LongTensor(input_ids)
                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                    scores = self.model(input_ids).detach().cpu().tolist()
                    # print('scores', scores)  # [0.00033855586661957204, 9.567374945618212e-05, 0.00014458030636887997, 3.3691056160023436e-05,....]
                #  模型推理：
                # self.model(input_ids) 执行前向传播，输出形状为 (batch_size, 2) 的 logits。
                # 分离计算图：
                # .detach() 将输出从计算图中分离，避免占用不必要的内存。
                # 移至 CPU：
                # .cpu() 将张量从 GPU 移回 CPU（若在 GPU 上）。
                # 转换为列表：
                # .tolist() 将张量转换为 Python 列表
                # np.argmax用于返回数组 scores 中最大值的索引
                hit_index = np.argmax(scores)
                # print(hit_index)
                predicts.append(hit_index)
            # print('predicts', predicts)  # [17, 17, 141, 735, 1465, 1627, 652, 168, 167, 652, 729, 140, 245...]
            self.write_stats(predicts, labels)
        self.show_stats()
        return

    def write_stats(self, predicts, labels):
        assert len(labels) == len(predicts)
        for hit_index, label in zip(predicts, labels):
            # question_index_to_standard_question_index中key为问题编号, value为属于哪一类
            hit_index = self.question_index_to_standard_question_index[hit_index]  # 转化成标准问编号
            if int(hit_index) == int(label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return
