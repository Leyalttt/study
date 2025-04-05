# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

"""
自回归语言模型训练
基于pytorch的LSTM语言模型, 预测下一个字
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        # print('input_dim, vocab', input_dim, vocab)  # 256 {'<pad>': 0, '<UNK>': 1, ' ': 2, '!': 3,
        super(LanguageModel, self).__init__()
        # vocab嵌入层可以处理词汇表中的所有单词, input_dim每个单词被映射到的向量
        self.embedding = nn.Embedding(len(vocab), input_dim)
        self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        # 映射成词表大小
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # output shape:(batch_size, sen_len, input_dim)
        x, _ = self.layer(x)  # output shape:(batch_size, sen_len, input_dim)
        y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
        # print('y_pred', y_pred.shape)  # torch.Size([64, 10, 3961])
        if y is not None:
            # print('y.shape0', y.shape)  #  torch.Size([64, 10])
            # print('y.view(-1)', y.view(-1).shape) # torch.Size([640])

            # print('y_pred.view', y_pred.shape)  # torch.Size([64, 10, 3961])
            # print('y_pred.view(-1, y_pred.shape[-1])', y_pred.view(-1, y_pred.shape[-1]).shape)  # torch.Size([640, 3961])
            """
            如果你只是想改变张量的形状，而不改变其数据，那么你应该使用 view 函数
            # y_pred.view(-1, y_pred.shape[-1]) 会将 y_pred 的形状改变为 (batch_size, input_dim)
            # 如果 y_pred 的形状是 (batch_size, seq_len, input_dim)，那么 y_pred.view(-1, y_pred.shape[-1]) 的形状就是 (batch_size * seq_len, input_dim)
            # y.view(-1) 会将 y 的形状改变为 (batch_size,)，这是为了将 y 的形状与 y_pred 的形状匹配，以便计算损失函数0
            -1都是自动算的
            """
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    # print('vocab_path', vocab_path)  # vocab.txt
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符, 倒数第二个字符 -1
            vocab[char] = index + 1  # 留出0位给pad token
    # print('vocab', vocab)  # {'<pad>': 0, '<UNK>': 1, ' ': 2, '!': 3, '(': 4, ')'
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    # print('corpus', corpus)
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    # 一个随机整数, 作为开始
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位, 因为预测下一个字
    # print('window', window)  # 斩向大阵，内外两方好
    # print('target', target)  # 向大阵，内外两方好不
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]  # 将字转换成序号
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    # print('x', x, 'y', y)  # x [2477, 2483, 3956, 2455, 342, 2099, 3564, 3956, 1336, 1336] y [2483, 3956, 2455, 342, 2099, 3564, 3956, 1336, 1336, 2430]
    return x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    # openings 输入
    # # vocab=》 {'<pad>': 0, '<UNK>': 1, ' ': 2, '!': 3, '(': 4, ')',
    # print('openings', openings)  # 让他在半年之前，就不能做出
    # 将vocab字典中的键和值进行互换
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    # print('reverse_vocab', reverse_vocab)  # {0: '<pad>', 1: '<UNK>', 2: ' ', 3: '!', 4: '(', 5: ')', 6: ','
    model.eval()
    # openings = "Hello, world!"
    window_size = 5
    # print(openings[-window_size:])  # 输出：world!
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            # 将openings字符串的最后window_size个字符转换为索引，并创建一个包含这些索引的torch.LongTensor张量x
            # -window_size表示从末尾开始计数，:表示取到末尾
            x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            # 0是因为只有一个batch, 就是openings自己, [-1]是取最后一个字
            # y是下一个字的概率分布
            y = model(x)[0][-1]
            # print('y', y)
            # 选中概率最大的index
            index = sampling_strategy(y)
            # 从词表中找出这个字, 加到openings后面
            pred_char = reverse_vocab[index]
    return openings


# 采样策略
def sampling_strategy(prob_distribution):
    # 生成一个0到1之间的随机浮点数
    if random.random() > 0.1:
        # 贪婪
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        # 选概率最高的
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        # 随机采样
        prob_distribution = prob_distribution.cpu().numpy()
        # np.random.choice将返回一个随机选择的索引
        # list(range(len(prob_distribution)))生成一个包含所有可能索引的列表
        # p=prob_distribution指定了每个索引的概率
        # print('list(range(len(prob_distribution)))', list(range(len(prob_distribution))))
        # print('np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)', np.random.choice(list(range(len(prob_distribution))), p=prob_distribution))
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


# 计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    # print('corpus_path', corpus_path)  # corpus.txt
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    char_dim = 256  # 每个字的维度
    window_size = 10  # 样本文本长度
    # vocab=》 {'<pad>': 0, '<UNK>': 1, ' ': 2, '!': 3, '(': 4, ')'
    vocab = build_vocab("vocab.txt")  # 建立字表
    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model(vocab, char_dim)  # 建立模型
    # print('model', model)
    # LanguageModel(
    #   (embedding): Embedding(3961, 256)
    #   (layer): LSTM(256, 256, batch_first=True)
    #   (classify): Linear(in_features=256, out_features=3961, bias=True)
    #   (dropout): Dropout(p=0.1, inplace=False)
    # )
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # 测试代码
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))  # 让他在半年之前，就不能做出一个人，说道：“你下来吧，你们也不知
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))  # 李慕站在山路上，深深的呼吸。李慕走到院子里，李慕和晚晚和小白一
        generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size)
        # print('ceshi')
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)
