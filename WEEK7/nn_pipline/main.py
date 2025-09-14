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

# level: 指定打印日志的级别,debug,info,warning,error,critical
# format: 日志输出相关格式
# %(asctime)s：日志记录的时间
# %(name)s：记录器的名称（通常是模块名）
# %(levelname)s：日志级别名称（如 INFO、WARNING 等）
# %(message)s：实际的日志消息内容
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# 2025-09-14 19:53:21,115 - __main__ - INFO - epoch 1 begin
# 2025-09-14 19:53:22,818 - __main__ - INFO - batch loss 3.219985

"""
模型训练主程序
"""

# 定义随机种子, 同样数据同样模型可以复现
# 从配置对象 Config 中获取一个名为 seed 的值，并将其赋给变量 seed。这个值通常是一个整数，用于初始化随机数生成器
seed = Config["seed"]
# 设置随机数生成器的种子。这样，每次运行程序时，random 模块生成的随机数序列都是相同的
random.seed(seed)  # Python 中的 random 模块, 用于生成伪随机数序列。以确保每次程序运行时都产生相同的随机数序列
np.random.seed(seed)  # NumPy 中的 numpy.random, 用于生成 NumPy 库中的随机数组。
torch.manual_seed(seed)  # 机器学习库中的随机性控制：比如 PyTorch 中的 torch.manual_seed(seed) 用于控制随机数生成，在模型训练中确保重复性。这个函数是用于CPU上的
torch.cuda.manual_seed_all(seed)  # 深度学习框架中的 GPU 随机性控制：对于使用 GPU 的深度学习框架（如 PyTorch、TensorFlow），通常也会有类似 torch.cuda.manual_seed(seed) 的函数，用于设置 GPU 相关的随机种子，以保证实验结果的一致性。
for i in range(3):
    random_number = random.random()
    print('random_number_{}:{}'.format(i, random_number))
    # 每次执行文件都会生成同样的数据
    # random_number_0:0.5327547554445934
    # random_number_1:0.5288157818367566
    # random_number_2:0.6265608260743084
random_tensor_cpu = torch.rand(3)
print("Random Tensor (CPU):", random_tensor_cpu)
# tensor([0.9281, 0.4066, 0.4132])
if torch.cuda.is_available():
    # 在GPU上生成随机张量
    for i in range(3):
        random_tensor_gpu = torch.rand(3).cuda()
        print("Random Tensor (GPU):", random_tensor_gpu)  # 一维数组


def main(config):
    # 创建保存训练好的模型的目录
    # os.path.isdir用于检查指定的路径是否是一个目录
    print('os.path.isdir(config["model_path"])', os.path.isdir(config["model_path"]))
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        # 是记录一条信息级别的日志，内容是 "gpu可以使用，迁移模型至gpu"
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
            if cuda_flag:
                # 这是一个列表推导式，用于将 batch_data 中的每个元素迁移到 GPU 上
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)

    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc


if __name__ == "__main__":
    main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    for model in ["gated_cnn", 'bert', 'lstm']:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        print("最后一轮准确率：", main(Config), "当前配置：", Config)
