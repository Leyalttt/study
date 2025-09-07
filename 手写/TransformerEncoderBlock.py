import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoderBlock(nn.Module):
    # embed_dim: 输入的维度, num_heads: 多头注意力机制中的头数, ffn_hidden_dim: 前馈神经网络隐藏层的维度, dropout: dropout 概率，默认为 0.1
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        # Multi-Head Self-Attention 多头注意力机制
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        # Layer normalization 归一化层
        self.n1 = nn.LayerNorm(embed_dim)
        self.n2 = nn.LayerNorm(embed_dim)
        # Feed-Forward Network (FNN)前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        # dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention with residual connection
        # 多头自注意力处理输入 x
        attention_output = self.self_attention(x, mask)
        # 然后将 attention_output 经过 dropout 处理后与原始输入 x 相加（残差连接）
        x = x + self.dropout(attention_output)  # residual
        # 最后通过层归一化处理结果
        x = self.n1(x)  # Layer normalization
        # Feed-Forward Network with residual connection
        # 将上一步的输出通过前馈神经网络处理，得到 ffn_output
        ffn_output = self.ffn(x)
        # 将 ffn_output 经过 dropout 处理后与输入相加（残差连接）
        x = x + self.dropout(ffn_output)  # residual connection
        # 最后通过第二个层归一化处理结果
        x = self.n2(x)  # Layer normalization
        return x
class MultiHeadAttention(torch.nn.Module):
    # hidden_size: 输入的维度, num_heads: 多头注意力机制中的头数, ffn_hidden_dim: 前馈神经网络隐藏层的维度, dropout: dropout 概率，默认为 0.1
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 这意味着我们将512维的特征空间分割成8个子空间，每个子空间64维
        # 初始化Q,K,V投影矩阵
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        # print('Q线性层权重形状', self.q_linear.weight.shape)  # torch.Size([512, 512])
        # print(f"Q线性层: 输入维度={self.q_linear.in_features}, 输出维度={self.q_linear.out_features}")  # Q线性层: 输入维度=512, 输出维度=512
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        # 输出线性层
        self.o_linear = nn.Linear(hidden_size, hidden_size)
        # dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_state, attention_mask=None):
        # 获取输入张量的批量大小（batch size），即第一个维度的大小。size()和shape功能相似
        batch_size = hidden_state.size()[0]
        query = self.q_linear(hidden_state)
        # print('query.shape', query.shape)  # torch.Size([2, 10, 512])
        # 输入hidden_state: [2, 10, 512]
        #                     |   |    |
        #                     |   |    └── 每个向量512维
        #                     |   └── 每个样本10个向量
        #                     └── 2个样本
        #
        # 应用self.q_linear (nn.Linear(512, 512))后:
        #                     |   |    |
        #                     |   |    └── 每个向量变换为512维
        #                     |   └── 每个样本仍然10个向量
        #                     └── 仍然2个样本
        #
        # 输出query: [2, 10, 512], Linear只对最后一层进行变换, 由输入512变为输出512
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)
        # split_head方法会将形状从[batch_size, seq_len, hidden_size]变为[batch_size, num_heads, seq_len, head_dim]
        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)
        # 计算注意力分数 行代码计算了查询（query）和键（key）之间的注意力分数，并进行了缩放处理
        # q * k的转置 / 根号batch_size
        # transpose(-1, -2) 最后一维和倒数第二维
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=query.dtype))
        if attention_mask is not None:
            # 如果掩码是3维的，添加一个头维度，使其与注意力分数的形状兼容
            if attention_mask.dim() == 3:
                # 如果是[batch_size, seq_len, seq_len]，则添加一个头维度
                attention_mask = attention_mask.unsqueeze(1)
            # masked_fill方法将注意力分数中掩码为0的位置的值设置为负无穷（float('-inf')）。
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
            # print('attention_scores', attention_scores) #[[ 3.2108e-01,  3.7669e-01,  1.9923e-01,  ..., -1.1387e-01,-1.6898e-01, -5.8413e-01],....]
            # print('attention_scores.shape', attention_scores.shape)  # torch.Size([2, 8, 10, 10])
        # 对注意力分数进行归一化
        # dim=-1：指定在哪个维度上应用softmax，-1表示最后一个维度
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # print('attention_probs.shape', attention_probs.shape)  # torch.Size([2, 8, 10, 10])
        # 应用dropout
        # softmax(q * k的转置 / 根号batch_size) * v
        output = torch.matmul(attention_probs, value)
        # print('output.shape', output.shape)  # torch.Size([2, 8, 10, 64])
        # 对注意力输出进行拼接
        # contiguous()：确保张量在内存中是连续存储的
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)
        # print('output.shape', output.shape)  # torch.Size([2, 10, 512])
        output = self.o_linear(output)
        return output

    def split_head(self, x):
        batch_size = x.size()[0]  # 2S
        # 这一步将输入张量x重塑为四维张量：
        # 原始输入x的形状是[2, 10, 512] 批量大小, 序列长度, 输入维度
        # 使用view方法将其重塑为[2, 10, 8, 64], 原始的512维特征被分割成8个64维的子空间，每个子空间对应一个注意力头,
        # 批量大小, 序列长度, 注意力头的数量, 每个头的维度

        # 原始输入张量x的形状是[2, 10, 512]，总元素数为2 * 10 * 512 = 10240。
        # x.view(2, -1, 8, 64) 所以-1是10,相乘才会等于10240
        # transpose(1, 2) 表示交换张量的第1维和第2维（注意：维度的索引从0开始）
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # (2, 8, 10, 64)
