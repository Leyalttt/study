import torch
import math
import numpy as np
from transformers import BertModel
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 几b就是几亿参数量
"""
通过了手动实现Bret结构
"""
# 预训练模型
# 加载了一个预训练的 BERT 模型(相当于别人帮我训练好了)
# from_pretrained用于加载预训练模型和分词器。这个函数可以帮助我们轻松地从模型库或本地路径加载已经训练好的模型
# # 原始字符串不会处理字符串中的反斜杠\作为转义字符
# return_dict=False 模型不以字典形式输出
bert = BertModel.from_pretrained(r"D:\TTT\NLP算法\bert-base-chinese\bert-base-chinese", return_dict=False)
# state_dict 是一个字典，包含了模型的所有参数（如权重和偏置）+ 参数名可遍历
state_dict = bert.state_dict()
bert.eval()
# 创建了一个包含四个字的一个句子的 NumPy 数组
x = np.array([2450, 15486, 102, 2110])  # 假想成4个字的句子
# 将 NumPy 数组转换为 PyTorch 张量
torch_x = torch.LongTensor([x])  # pytorch形式输入
# 将张量传递给 BERT 模型，并获取模型的输出。seqence_output 是经过12层的k 用于预测词表中真是的字, 输入文本长度(4) * 768
# pooler_output 是池化输出, 是每句话的cls token对应的向量在通过线性层,用于给整句话分类的 1*768(句子的嵌入表示 一句话所以是1*768)
seqence_output, pooler_output = bert(torch_x)
# print(seqence_output.shape, pooler_output.shape)
# torch.Size([1, 4, 768])第一个维度是批次大小/句子数量 （batch size），第二个维度是序列长度（sequence length），第三个维度是隐藏层大小（hidden size）
# torch.Size([1, 768]) 第一个维度是是句子的数量，第二个维度是隐藏层大小
# print('seqence_output', seqence_output, 'pooler_output', pooler_output)
# 单一的固定长度的向量来表示整个句子, 后续被用作分类层的输入, cls最后一层的隐藏状态在进行线性层和tanh激活函数得到pooler_output
"""
seqence_output tensor([[[ 0.0169,  0.0160, -0.5747,  ..., -0.1830,  0.1512, -0.0556],
         [-1.2614,  0.6010, -0.7746,  ..., -0.5891, -0.2752,  0.1237],
         [-0.2093,  0.6627, -0.3288,  ..., -1.0358, -0.2932,  0.5032],
         [-0.7496, -0.1591, -0.4067,  ..., -0.6937,  0.1669,  1.3460]]],
         grad_fn=<NativeLayerNormBackward0>) 
pooler_output tensor([[-4.6565e-01,  5.9548e-01, -9.1259e-01,  5.6502e-01,  4.3504e-01,
          6.9858e-01, -5.2608e-01, -6.5987e-01,  7.2654e-01, -3.7667e-01,
          5.6364e-01, -9.6316e-01, -4.4638e-01, -3.7205e-01,  8.4157e-01,
         -4.2996e-01, -5.0768e-01,  8.3469e-01, -3.7882e-01,  7.7538e-01,
          8.8424e-01, -4.0783e-01, -8.1155e-01,  6.9903e-01,  8.0690e-01,
          4.7617e-01, -3.1287e-01, -3.1466e-01, -9.8350e-01, -6.9434e-01,
         -1.1137e-01,  5.4048e-01, .....  4.4426e-01]], grad_fn=<TanhBackward0>)
"""

# print(bert.state_dict().keys())  #查看所有的权值矩阵名称
"""
# odict_keys(['embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight',
# 'embeddings.token_type_embeddings.weight', 'embeddings.LayerNorm.weight', 'embeddings.LayerNorm.bias',
# 'encoder.layer.0.attention.self.query.weight', 'encoder.layer.0.attention.self.query.bias',
# 'encoder.layer.0.attention.self.key.weight', 'encoder.layer.0.attention.self.key.bias',
# 'encoder.layer.0.attention.self.value.weight', 'encoder.layer.0.attention.self.value.bias',
# 'encoder.layer.0.attention.output.dense.weight', 'encoder.layer.0.attention.output.dense.bias',
# 'encoder.layer.0.attention.output.LayerNorm.weight', 'encoder.layer.0.attention.output.LayerNorm.bias',
# 'encoder.layer.0.intermediate.dense.weight', 'encoder.layer.0.intermediate.dense.bias',
# 'encoder.layer.0.output.dense.weight', 'encoder.layer.0.output.dense.bias',
# 'encoder.layer.0.output.LayerNorm.weight',
# 'encoder.layer.0.output.LayerNorm.bias', 'pooler.dense.weight', 'pooler.dense.bias'])
"""


# 手动实现
# softmax归一化
# 输入的实数向量转换为概率分布，使得所有元素都在 0 和 1 之间，并且所有元素的和为 1。这样，每个元素 y_i 可以被解释为输入属于第 i 类的概率。
def softmax(x):
    # exp(x) 是 e 的 x 次方，sum(exp(x)) 是向量 x 中所有元素 e 的 x 次方的和
    # axis=-1 表示沿着最后一个轴（通常是列）进行求和，keepdims=True 表示保持结果的维度
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


# gelu激活函数
def gelu(x):
    # math.sqrt(2 / math.pi) 计算一个平方根 /....044715 * np.power(x, 3) 计算了 0.044715 乘以 x 的三次方，
    # np.tanh(...) 计算了双曲正切函数，0.5 * x * (1 + np.tanh(...)) 计算了 GELU 函数的值。
    # 不理解就直接抄论文里就是这个公式
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))


# 初始化模型
class DiyBert:
    # 将预训练好的整个权重字典输入进来
    def __init__(self, state_dict):
        self.num_attention_heads = 12  # 12个头
        self.hidden_size = 768  # 维度768
        self.num_layers = 1  # 注意这里的层数要跟预训练config.json文件中的模型层数一致
        self.load_weights(state_dict)

    # 将预训练权重加载到模型中
    def load_weights(self, state_dict):
        # embedding部分
        # 从 state_dict 中获取词嵌入层的权重，并将其转换为 NumPy 数组。
        # 这行代码从state_dict中获取词嵌入层的权重，并将其转换为NumPy数组。state_dict是一个字典，包含了BERT模型的权重。
        # "embeddings.word_embeddings.weight"是词嵌入层的权重在state_dict中的键
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()  # 词表大小*768
        # print('state_dict["embeddings.word_embeddings.weight"]', state_dict["embeddings.word_embeddings.weight"])
        # 张量 tensor([[]])
        # tensor([[ 0.0262,  0.0109, -0.0187,  ...,  0.0903,  0.0028,  0.0064],
        #         ...,
        #         [ 0.0346,  0.0021,  0.0085,  ...,  0.0085,  0.0337,  0.0099],
        #         [ 0.0541,  0.0289,  0.0263,  ...,  0.0526,  0.0651,  0.0353],
        #         [ 0.0200,  0.0023, -0.0089,  ...,  0.0799, -0.0562,  0.0247]])
        # print('state_dict["embeddings.word_embeddings.weight"].numpy()', state_dict["embeddings.word_embeddings.weight"].numpy())
        # 转为numpy [[]]
        # [[ 0.02615827  0.01094903 -0.01868878 ...  0.09030139  0.0028486
        #    0.00642775]
        #  ...
        #  [ 0.05406349  0.02890619  0.02626012 ...  0.0525924   0.06508742
        #    0.03532186]
        #  [ 0.02002425  0.00229523 -0.00892451 ...  0.07987329 -0.05615233
        #    0.02471835]]
        # 从state_dict中获取位置嵌入层的权重，并将其转换为NumPy数组。位置嵌入层的权重在state_dict中的键是"embeddings.position_embeddings.weight"
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()  # 最大长度512 765*512
        # 从state_dict中获取token type嵌入层的权重 用于区分句子，并将其转换为NumPy数组。token type嵌入层的权重在state_dict中的键是"embeddings.token_type_embeddings.weight"
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        # print('token_type_embeddings', self.token_type_embeddings.shape)  # (2, 768)
        # state_dict中获取嵌入层的LayerNorm层的权重，并将其转换为NumPy数组。嵌入层的LayerNorm层的权重在state_dict中的键是"embeddings.LayerNorm.weight"
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        # 从state_dict中获取嵌入层的LayerNorm层的偏置，并将其转换为NumPy数组。嵌入层的LayerNorm层的偏置在state_dict中的键是"embeddings.LayerNorm.bias"
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        # 循环transformer部分，有多层, num_layers之前定义1层
        for i in range(self.num_layers):
            # 从 state_dict 中获取每一层的权重，并将其转换为 NumPy 数组。这些权重包括查询q、k、v的权重和偏置，
            # 注意力输出层的权重和偏置，注意力层归一化层的权重和偏置，前馈神经网络层的权重和偏置，以及前馈神经网络层归一化层的权重和偏置
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            # 注意力机制输出部分的线性变换层的权重矩阵和偏置向量
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            # 意力机制输出部分的层归一化层的权重矩阵和偏置向量
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            # 馈神经网络部分的线性变换层的权重矩阵和偏置向量
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            # print('intermediate_weight', intermediate_weight.shape)  # (3072, 768)
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            # 前馈神经网络部分的输出部分的线性变换层的权重矩阵和偏置向量
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            # 前馈神经网络部分的输出部分的层归一化层的权重矩阵和偏置向量
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            self.transformer_weights.append(
                [q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                 attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                 output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        # pooler层?
        # 从 state_dict 中获取池化层的权重和偏置，并将其转换为 NumPy 数组
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

    # BERT模型的前向传播
    # 这个函数的主要作用是计算输入文本的词嵌入表示，并进行归一化处理
    # bert embedding，使用3层叠加，在经过一个Layer norm层
    def embedding_forward(self, x):
        # print('x.shap', x.shape)  # x.shap (4,)
        # x.shape = [max_len]
        # 获取输入文本的词嵌入
        # self.word_embeddings 是词嵌入矩阵，x 是输入文本的索引序列。self.get_embedding 函数用于将索引序列转换为词嵌入表示
        # self 是 类 的一个实例
        we = self.get_embedding(self.word_embeddings,
                                x)  # shpae: [max_len(4), hidden_size(768)]  x = [ 2450 15486   102  2110]
        # position embeding的输入 [0, 1, 2, 3] np.array(list(range(len(x))))  创建一个从0到len(x)-1的整数数组
        pe = self.get_embedding(self.position_embeddings,
                                np.array(list(range(len(x)))))  # shpae: [max_len, hidden_size]
        # token type embedding,单输入的情况下为[0, 0, 0, 0]这些词元都属于同一部分  [0] * len(x) 创建了一个包含 len(x) 个0的列表
        te = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))  # shpae: [max_len, hidden_size]
        embedding = we + pe + te
        # 加和后有一个归一化层
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight,
                                    self.embeddings_layer_norm_bias)  # shpae: [max_len, hidden_size]
        return embedding

    # embedding层实际上相当于按index索引
    def get_embedding(self, embedding_matrix, x):
        # print('embedding_matrix.shape', embedding_matrix.shape)
        # (21128, 768)  (512, 768)  (2, 768)
        return np.array([embedding_matrix[index] for index in x])

    # 执行全部的transformer层计算
    # BERT 模型的 Transformer 编码器层的前向传播函数
    # 作用是依次执行所有的 Transformer 编码器层
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    # 执行单层transformer层计算
    # 主要作用是执行单层 Transformer 编码器层的计算，包括自注意力机制、前馈神经网络和层归一化等操作。
    def single_transformer_layer_forward(self, x, layer_index):
        # 从 self.transformer_weights 中取出当前层的权重。self.transformer_weights 是一个列表，包含了所有层的权重
        weights = self.transformer_weights[layer_index]
        # 取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        q_w, q_b, \
            k_w, k_b, \
            v_w, v_b, \
            attention_output_weight, attention_output_bias, \
            attention_layer_norm_w, attention_layer_norm_b, \
            intermediate_weight, intermediate_bias, \
            output_weight, output_bias, \
            ff_layer_norm_w, ff_layer_norm_b = weights
        # self attention层
        attention_output = self.self_attention(x,
                                               q_w, q_b,
                                               k_w, k_b,
                                               v_w, v_b,
                                               attention_output_weight, attention_output_bias,
                                               self.num_attention_heads,
                                               self.hidden_size)
        # bn层，并使用了残差机制
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        # feed forward层
        feed_forward_x = self.feed_forward(x,
                                           intermediate_weight, intermediate_bias,
                                           output_weight, output_bias)
        # bn层，并使用了残差机制
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x

    # self attention的计算
    def self_attention(self,
                       x,
                       q_w,
                       q_b,
                       k_w,
                       k_b,
                       v_w,
                       v_b,
                       attention_output_weight,
                       attention_output_bias,
                       num_attention_heads,
                       hidden_size):
        # x.shape = max_len * hidden_size
        # q_w, k_w, v_w  shape = hidden_size * hidden_size
        # q_b, k_b, v_b  shape = hidden_size
        # q_w 768 * 768 q_b 768
        q = np.dot(x, q_w.T) + q_b  # shape: [max_len, hidden_size]      W * X + B lINER
        k = np.dot(x, k_w.T) + k_b  # shpae: [max_len, hidden_size]
        v = np.dot(x, v_w.T) + v_b  # shpae: [max_len, hidden_size]
        attention_head_size = int(hidden_size / num_attention_heads)
        # q.shape = num_attention_heads, max_len, attention_head_size
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        # k.shape = num_attention_heads, max_len, attention_head_size
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        # v.shape = num_attention_heads, max_len, attention_head_size
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)
        # qk.shape = num_attention_heads, max_len, max_len
        qk = np.matmul(q, k.swapaxes(1, 2))
        qk /= np.sqrt(attention_head_size)
        qk = softmax(qk)
        # qkv.shape = num_attention_heads, max_len, attention_head_size
        qkv = np.matmul(qk, v)
        # qkv.shape = max_len, hidden_size
        # .reshape(-1, hidden_size) -1是自动算出来的
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        # attention.shape = max_len, hidden_size
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention

    # 多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        # hidden_size = 768  num_attent_heads = 12 attention_head_size = 64
        max_len, hidden_size = x.shape
        # x.reshape() 是 NumPy 数组的一个方法，用于改变数组的形状，而不改变其数据。
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 0)  # output shape = [num_attention_heads, max_len, attention_head_size]
        return x

    # 前馈网络的计算
    def feed_forward(self,
                     x,
                     intermediate_weight,  # intermediate_size, hidden_size
                     intermediate_bias,  # intermediate_size
                     output_weight,  # hidden_size, intermediate_size
                     output_bias,  # hidden_size
                     ):
        # output shpae: [max_len, intermediate_size]
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        # output shpae: [max_len, hidden_size]
        x = np.dot(x, output_weight.T) + output_bias
        return x

    # 归一化层固定公式
    # 首先计算输入 x 的均值和方差，然后通过缩放和平移操作将其归一化到均值为0，方差为1的标准正态分布。
    # 最后，将归一化后的 x 与权重 w 和偏置 b 进行线性变换，得到最终的输出
    def layer_norm(self, x, w, b):
        # 计算输入 x 的均值和方差，然后通过缩放和平移操作将其归一化到均值为0，方差为1的标准正态分布
        # np.mean(x, axis=1, keepdims=True) 计算的是 x 在第1维（即每个样本的特征维度）上的均值，
        # np.std(x, axis=1, keepdims=True) np.std(x, axis=1, keepdims=True) 计算的是 x 在第1维上的标准差。
        # axis=1 表示沿着序列长度维度计算均值和标准差。这是因为层归一化是对每个样本的每个词元进行归一化，而不是对整个批次进行归一化。
        # 具体来说，np.mean(x, axis=1, keepdims=True) 计算的是每个样本在每个词元上的均值，np.std(x, axis=1, keepdims=True)
        # 计算的是每个样本在每个词元上的标准差。这样，对于每个样本的每个词元，都可以通过减去均值和除以标准差来进行归一化
        # print('x', x, 'np.mean(x, axis=1, keepdims=True)', np.mean(x, axis=1, keepdims=True))
        # [[-0.01156412  0.07873195 -1.1277188  ... -0.7492888   0.2562669
        #   -0.16318391]
        #  [-2.2807841   1.1248959  -1.3314021  ... -1.5861504  -0.55319774
        #    0.20647597]
        #  [-0.41326046  1.1583594  -0.5374819  ... -2.5590363  -0.5947111
        #    0.84705   ]
        #  [-1.0431904  -0.18875994 -0.52065665 ... -1.3911612   0.18157119
        #    1.8327646 ]]
        # [[-0.02371312]
        #  [-0.00398928]
        #  [-0.03391678]
        #  [-0.02425162]]
        # 比如:
        # x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        #               [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
        # np.mean(x, axis=1, keepdims=True) = 取出第一个样本的第一个词元：[1, 2, 3]
        # 然后，计算这个词元的均值：(1 + 2 + 3) / 3 = 2 以此类推
        # 结果为:[[[2. 5. 8.]]
        #  [[11. 14. 17.]]]
        # 均方差
        # x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        #               [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
        # 首先，计算这个词元中每个元素与均值的差的平方：[(1-2)^2, (2-2)^2, (3-2)^2] = [1, 0, 1],  = 0.33333 开根号为 0.5773502691896257
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    # 链接[cls] token的输出层https://github.com/Leyalttt/study/tree/main
    # 这个函数的主要作用是从序列的最后一个隐藏状态中提取一个固定长度的向量表示
    def pooler_output_layer(self, x):
        # 将输入 x 与池化层的权重 self.pooler_dense_weight 进行矩阵乘法，然后加上偏置 self.pooler_dense_bias，得到中间结果
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        # 对中间结果应用 tanh 激活函数
        x = np.tanh(x)
        # print('x.shape', x.shape)  # x.shape (768,)
        return x

    # 最终输出
    def forward(self, x):
        # print('x', x)  # [ 2450 15486   102  2110]
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        # print('sequence_output', sequence_output.shape)  # (4, 768)  其中包含4行和768列
        pooler_output = self.pooler_output_layer(sequence_output[0]) # 返回第一行
        return sequence_output, pooler_output


# 自制
db = DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)
# torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

print(diy_sequence_output.shape)  # (4, 768)
print(torch_sequence_output.shape)  # torch.Size([1, 4, 768])

print(diy_pooler_output.shape)  # (768,)
print(torch_pooler_output.shape)  # torch.Size([1, 768])
