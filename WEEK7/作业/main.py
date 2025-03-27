# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

"""
在提供的数据集上训练文本分类模型，对比不同模型效果
任务复杂bert越好
"""

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
# 日志记录的基本设置。level=logging.INFO 表示只记录 INFO 级别及以上的日志消息。format 参数指定了日志消息的格式，
# '%(asctime)s - %(name)s - %(levelname)s - %(message)s' 表示日志消息应该包含时间戳、记录器名称、日志级别和日志消息本身
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 这行代码创建了一个名为 __name__ 的记录器
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

# # 定义随机种子, 同样数据同样模型可以复现
# # 目的是在多个不同的随机数生成器中设置相同的种子，以确保在运行程序时生成的随机数序列是一致的
# # 从配置对象 Config 中获取一个名为 seed 的值，并将其赋给变量 seed。这个值通常是一个整数，用于初始化随机数生成器
seed = Config["seed"]
# # 设置随机数生成器的种子。这样，每次运行程序时，random 模块生成的随机数序列都是相同的
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # print('config', config)
    # # os.path.isdir用于检查指定的路径是否是一个目录 不是就返回函数创建一个名为 config["model_path"] 的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # print('train_data', train_data)  # <torch.utils.data.dataloader.DataLoader object at 0x000001ADC7518470>
    # print('len(train_data)', len(train_data))  # 800, 每个 epoch 中 9589个句子/batch_size 每个批次的大小12 => 800个批次(batch)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    print('cuda_flag', cuda_flag)
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    # 每个epoch中的训练数据是相同的, 如果启用了 shuffle=True，则每个 epoch 开始时，数据会被打乱顺序，但数据内容不变。
    # 在每个 batch 中模型根据当前 batch 的数据计算前向传播的输出。计算损失（loss）。通过反向传播计算梯度。使用优化器（如 SGD、Adam）更新模型参数
    # 每个 epoch 中，模型参数会更新 batch(800) 次
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        # epoch 1 begin => epoch会替代%d
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        #     799                              800个批次
        for index, batch_data in enumerate(train_data):
            # print('batch_data', type(batch_data))  # list类型 -> loader.py中self.data.append([input_id, label_index])
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            # print('input_ids', input_ids.shape)  # torch.Size([12, 30])
            # print('labels', labels) # tensor([[0],[1],[0],[0],[0],[0],[0],[1],[0],[0],[1],[0]])  12个
            #               x         y
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            # 在训练过程中，每隔一定数量的 batch，记录当前 batch 的损失值
            if index % int(len(train_data) / 2) == 0:
                # print('index', index) # 0 400 800 每轮记录三次
                # batch loss 0.743423 => loss会代替%f
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)

    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc


if __name__ == "__main__":
    # print('Config["model_type"]', Config["model_type"])  # bert
    main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print('model', model)
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索,结果写入excel
    import pandas as pd
    # 用于存储每个超参数组合的准确率, columns列名是参数
    df = pd.DataFrame(columns=["model_type", "learning_rate", "hidden_size", "batch_size", "pooling_style", "acc"])
    # 通过嵌套循环，可以遍历所有超参数的可能组合
    for model in ["gated_cnn", 'fast_text', 'lstm']:
        # 模型类型
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            # 学习率
            Config["learning_rate"] = lr
            for hidden_size in [128, 256]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 256]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        acc = main(Config)
                        print('acc', acc)
                        # 将当前超参数组合和准确率添加到 DataFrame 中
                        df = df._append({"model_type": model, "learning_rate": lr, "hidden_size": hidden_size, "batch_size": batch_size, "pooling_style": pooling_style, "acc": acc}, ignore_index=True)
    # to_excel：将 DataFrame 保存为 Excel 文件。index=False：不保存行索引
    df.to_excel("result.xlsx", index=False)

