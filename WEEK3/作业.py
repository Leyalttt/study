# coding:utf8
import random
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
输入一个字符串，根据字符a所在位置进行分类
对比rnn和pooling做法

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # embedding层
        # pool会失去语序信息就不能得到它的位置
        # self.pool = nn.AvgPool1d(sentence_length)   #池化层
        # 可以自行尝试切换使用rnn
        # vector_dim输入特征和vector_dim隐藏状态的维度, 40*10*30 batch_size(每次训练样本个数) * sentence_length(文本长度) * char_dim(每个字的维度)
        # 如果 batch_first=False（默认），输入形状为 (sequence_length, batch_size, input_size)
        # batch_first=True：指定输入数据的形状为 (batch_size, sequence_length, input_size)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)

        # +1的原因是可能出现a不存在的情况，那时的真实label在构造数据时设为了sentence_length
        # 映射的值取决于想要什么, a所在位置进行分类, 所以每次都取sentence_length个, 再加一个啊不存在
        self.classify = nn.Linear(vector_dim, sentence_length + 1)     
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值, x是输入的向量
    def forward(self, x, y=None):
        # print(x.shape)  # 训练torch.Size([40, 10]), 测试torch.Size([200, 10]), 预测torch.Size([4, 10])
        x = self.embedding(x)
        # print(x.shape)  # 训练torch.Size([40, 10, 30]), 测试torch.Size([200, 10, 30]), 预测torch.Size([4, 10, 30])


        # nn.RNN 的返回值是一个元组 (output, hidden)
        # 使用rnn的情况 rnn_out 每个时间步的隐藏状态(预测结果/结果分类), 最后一个时间步的隐藏状态hidden(包含了在每个时间步的隐藏状态)
        # batch_first为false的形状 sentence_length(文本长度) * batch_size(每次训练样本个数)  * char_dim(每个字的维度)
        # batch_first为true的形状  batch_size(每次训练样本个数) * sentence_length(文本长度) * char_dim(每个字的维度)
        rnn_out, hidden = self.rnn(x)
        # print('rnn_out', rnn_out)
        # print('rnn_out', rnn_out.shape) # 40*10*30 batch_size(每次训练样本个数) * sentence_length(文本长度) * char_dim(每个字的维度)
        # print('hidden', hidden.shape)  # 1*40*30 num_layers(RNN包含隐藏层数不设置就是1) * batch_size(每次训练样本个数) * char_dim  在最后一个时间步的隐藏状态， 它携带了序列的“历史记忆”
        # 去掉值为中间文本长度的维度 提取每个样本在最后一个时间步的隐藏状态
        x = rnn_out[:, -1, :]  # 或者写hidden.squeeze()也是可以的，因为rnn的hidden就是最后一个位置的输出
        # print('x', x)
        # print('x', x.shape)  # 40*30 batch_size * char_dim(每个字的维度)
        # 接线性层做分类
        y_pred = self.classify(x)
        """训练40*11 测试200*11 预测: 4* 11"""
        # print('y_pred', y_pred.shape)
        if y is not None:
            return self.loss(y_pred, y)   # 预测值和真实值计算损失
        else:
            # print("y_pred", y_pred)
            return y_pred                 # 输出预测结果


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijk"  # 字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 12
    # print('vocab', vocab)  # {'pad': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'unk': 12}
    return vocab


# 随机生成一个样本
# sentence_length 样本文本长度
def build_sample(vocab, sentence_length):
    #  vocab 词汇表中keys随机选择 sentence_length 个不重复的key，并返回这些单词的列表
    # print('list(vocab.keys()', list(vocab.keys()))  # ['pad', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'unk']
    x = random.sample(list(vocab.keys()), sentence_length)
    # print(x)  # 比如:['c', 'j', 'unk', 'b', 'i', 'g', 'pad', 'k', 'h', 'e']

    # 指定哪些字出现时为正样本
    # 字符串x中是否包含字符"a"。如果x中包含字符"a"，那么y将被赋值为"a"在x中的索引
    if "a" in x:
        y = x.index("a")  # 下标应该是0-sentence_length-1(0-9)
    else:
        y = sentence_length  # 每次都是选择sentence_length(样本长度)个所以y没找到就是sentence_length最大长度 (10)
    x = [vocab.get(word, vocab['unk']) for word in x]   # 将字转换成序号，为了做embedding, 字母汉字embedding也看不懂
    # print('x, y', x, y)  # [5, 0, 4, 12, 7, 2, 9, 3, 11, 6] 10  y为10代表随机取得数没有a
    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
# sample_length 样本训练个数， sentence_length文本长度
def build_dataset(batch_size, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(batch_size):
        # 一个样本
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    # print('type', type(dataset_y))   # <class 'list'>
    # sample_length个样本数据拼接在一起
    # print('x', torch.LongTensor(dataset_x).shape, 'y',torch.LongTensor(dataset_y).shape)  # torch.Size([40, 10])  torch.Size([40])
    # 要将x，y转成张量后续model传参需要张量
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    # print('sample_length', sample_length)  # 10
    model.eval()
    # 200每次样本训练个数， 10样本长度
    x, y = build_dataset(200, vocab, sample_length)   # 建立200个用于测试的样本
    # print("本次预测集中共有%d个样本" %(len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        #  y_pred = model(x)放在with torch.no_grad()内 不会计算梯度,省内存
        # print('type', type(x))  # <class 'torch.Tensor'>
        y_pred = model(x)      # 模型预测
        # print('y_pred', y_pred.shape)  # torch.Size([200, 11])
        # 在forward里 y_pred = self.classify(x) 的形状就是batch_size*sentence_length + 1
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # torch.argmax(y_p)找到11个值中最大值的索引
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    # print("正确预测个数：%d, 正确率：%f" %(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    # 配置参数
    epoch_num = 20        # 训练轮数
    batch_size = 40       # 每次训练样本个数
    train_sample = 1000    # 每轮训练总共训练的样本总数
    char_dim = 30         # 每个字的维度
    sentence_length = 10   # 样本文本长度
    learning_rate = 0.001  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            # 思路: 求loass需要x,y, 所以需要调用build_dataset求x,y
            # loss.backward()计算梯度不更新参数, optim.step() 更新参数, optim.zero_gard() 梯度为0 后面需要求平均梯度
            x, y = build_dataset(batch_size, vocab, sentence_length) # 构造一组训练样本
            optim.zero_grad()    # 梯度归零
            # model中的参数为张量
            loss = model(x, y)   # 计算loss
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item()) # loss.item() 将张量转成标量
        # print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   # 测试本轮模型结果
        # 将这个包含两个元素的列表 [acc, np.mean(watch_loss)] 添加到 log 列表的末尾
        log.append([acc, np.mean(watch_loss)])

    # 画图
    # 参数1. x轴, 参数2y轴, 参数3是标签
    # range(len(log))：生成一个从0到len(log) - 1的整数序列，表示训练轮次。len(log)返回log列表的长度，即训练的总轮数。
    # [l[0] for l in log]：这是一个列表推导式，用于从log列表中提取每个元素（即每个训练轮次的结果）的第一个值（l[0]）
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    # plt.show()
    # 保存模型到"model.pth"中
    # model.state_dict()`返回的是一个有序字典OrderedDict，**该有序字典中保存了模型所有参数的参数名和具体的参数值，
    # 所有参数包括可学习参数和不可学习参数，可通过循环迭代打印参数**, 因此，该方法可用于保存模型，当保存模型时，
    # 会将不可学习参数也存下，当加载模型时，也会将不可学习参数进行赋值。
    torch.save(model.state_dict(), "model.pth")

    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    # json.dumps() 用于将 Python 对象编码成 JSON 字符串, 包含原始的非 ASCII 字符, 有两个空格
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    # print('input_strings', input_strings)  #  ["kijabcdefh", "gijkbcdeaf", "gkijadfbec", "kijhdefacb"]
    char_dim = 30  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载json格式字符表
    model = build_model(vocab, char_dim, sentence_length)     # 建立模型
    model.load_state_dict(torch.load(model_path, weights_only=False)) # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()   # 测试模式
    with torch.no_grad():  # 不计算梯度
        # 参数是张量
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i])) #打印结果


if __name__ == "__main__":
    main()
    test_strings = ["kijabcdefh", "gijkbcdeaf", "gkijadfbec", "kijhdefacb"]
    predict("model.pth", "vocab.json", test_strings)
