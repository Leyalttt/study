# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        # print('vocab', vocab)
        # {'pad': 0, '你': 1, '我': 2, '他': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
        # 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21,
        # 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, 'unk': 27}

        # print(vector_dim, len(vocab))  # 20, 28
        # len(vocab)：词汇表大小, vector_dim：每个字符的向量维度, padding_idx=0：指定padding字符的索引为0，这些字符不会被训练
        # 词汇表有28 (len(vocab))个词，每个词都会被表示成一个20(vector_dim)维的向量
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        # sentence_length 样本文本长度m 将整个序列压缩成一个单一向量
        self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        # 映射的值取决于想要什么, 是否有某些特定字符出现, 出现或不出现 所以每次都取1个， 或者要预测下个字是什么就映射为词典的大小
        self.classify = nn.Linear(vector_dim, 1)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # print('x.shape', x.shape)  # 训练torch.Size([20, 6]), 测试:torch.Size([200, 6])
        x = self.embedding(x)
        # 每个批次中的每一个索引都会被表示成一个vecter_dim维的向量
        # print(x.shape)  # 训练torch.Size([20, 6, 20]), 测试:torch.Size([200, 6, 20])
        x = x.transpose(1, 2)  # pooling默认对最后一维进行pooling所以12维交换

        x = self.pool(x)
        # print(x.shape) # 训练:orch.Size([20, 20, 1]), 测试:torch.Size([200, 20, 1])
        x = x.squeeze()  # 去除维度维1的
        x = self.classify(x)
        # print('x', x.shape)  # 训练:torch.Size([20, 1]), 测试:torch.Size([200, 1])
        y_pred = self.activation(x)
        # print('y_pred', y_pred.shape)  # # 训练:torch.Size([20, 1]), 测试:torch.Size([200, 1])
        if y is not None:
            # 训练走这里
            return self.loss(y_pred, y)  # 预测值和真实值计算损失, 维度要相同 torch.Size([20, 1])
        else:
            # 预测走这里
            # print('y_pred', y_pred.shape)  # torch.Size([200, 1])
            return y_pred  # 输出测试结果


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 指定哪些字出现时为正样本
    # x 中是否包含特定字符 "你我他" 中的任意一个或多个字符
    if set("你我他") & set(x):
        y = 1
    # 指定字都未出现，则为负样本
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(batch_size, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    # 每个batch
    for i in range(batch_size):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        # 为什么y外面要加 [], 确保最终的 dataset_y 是一个二维张量, 和y值的维度相同
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    # print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    # print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表, dict形式的
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    # epoch_num轮
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        #  500总数 / batch_size个  500 / 20 = 25=> 也就是训练 25个20(batch) 就是500了
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            # print('x, y', x.shape, y.shape)  # torch.Size([20, 6]) torch.Size([20, 1])
            # 每个batch都要梯度归零, 计算loss, 计算梯度, 更新权重
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    # plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        # round四舍五入
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i]))  # 打印结果
        # 输入：fnvfee, 预测类别：0, 概率值：0.112250


# 输入：fnvfee, 预测类别：0, 概率值：0.127793
# 输入：wz你dfg, 预测类别：1, 概率值：0.762213
# 输入：rqwdeg, 预测类别：0, 概率值：0.131950
# 输入：n我kwww, 预测类别：1, 概率值：0.818050

if __name__ == "__main__":
    main()
    test_strings = ["fnvfee", "wz你dfg", "rqwdeg", "n我kwww"]
    predict("model.pth", "vocab.json", test_strings)
