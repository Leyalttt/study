# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import SentenceMatchNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
交互性文本匹配
模型训练主程序
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # print('len(train_data)', len(train_data))  # 79 一个 epoch 中需要迭代的批次数
    # 加载模型
    model = SentenceMatchNetwork(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:  # 如果gpu可用则使用gpu加速
                batch_data = [d.cuda() for d in batch_data]
            # input_ids =>loader.py中__getitem__方法生成俩个拼接在一起的id, labels 1是正样本, 0是负样本
            input_ids, labels = batch_data
            # print('input_ids', input_ids.shape)  # torch.Size([128, 20])
            # print('labels', labels.shape)  # torch.Size([128, 1])
            # 128：当前批次的样本数量(一次性处理 128 个样本)。20：每个样本的编码长度（由 max_length 参数控制）。
            loss = model(input_ids, labels)  # 计算loss
            train_loss.append(loss.item())
            # 每轮训练一半的时候输出一下loss，观察下降情况
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
            loss.backward()  # 梯度计算
            optimizer.step()  # 梯度更新
        logger.info("epoch average loss: %f" % np.mean(train_loss))
    evaluator.eval(config["epoch"])
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)
    return


if __name__ == "__main__":
    main(Config)
