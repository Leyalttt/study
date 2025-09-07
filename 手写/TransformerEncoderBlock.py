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
        # 先的放大四倍, 再缩小
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
        # 多头自注意力处理输入 x (就是q)
        attention_output = self.self_attention(x, mask)
        # print('attention_output.shape', attention_output.shape)  # torch.Size([2, 10, 512])
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


# 2==============================================
# 交叉注意力(Cross-Attention)机制, 也称为编码器-解码器注意力
class EncoderDecoderAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super(EncoderDecoderAttention, self).__init__()
        # 这个Multi-HeadAttention forward的时候接受QKV对应的输入
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads, dropout)

    def forward(self, x, encoder_output, mask=None):
        # Calculate attention based on encoder's output(cross-attention)
        # KV:encoder_output编码器的输出，Q: X
        # 第一个参数 x 充当 Query (Q) 的角色 - 这来自解码器的输入，代表"我想查询什么"
        # 第二个参数 encoder_output 充当 Key (K) 的角色 - 这来自编码器的输出，代表"用来匹配的键"
        # 第三个参数 encoder_output 充当 Value (V) 的角色 - 同样来自编码器的输出，代表"匹配成功后获取的值"
        # 第四个参数 mask 是可选的注意力掩码
        attention_output = self.cross_attention(hidden_state=x, k=encoder_output, v=encoder_output, attention_mask=mask)

        return attention_output


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        # masked Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.encoder_decoder_attention = EncoderDecoderAttention(embed_dim, num_heads, dropout)
        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        # Feed-Forward Network(FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        # Self-Attention with residual connection
        # 执行函数EncoderDecoderAttention.forward
        self_attention_output = self.self_attention(hidden_state=x, k=None, v=None, attention_mask=self_attention_mask)
        # print('self_attention_output.shape', self_attention_output.shape)  # torch.Size([2, 10, 512])
        x = x + self.dropout(self_attention_output)  # Residual connection
        x = self.ln1(x)  # Layer normalization
        # Encoder-Decoder Attention(Cross-Attention) with residual connection
        encoder_decoder_output = self.encoder_decoder_attention(x, encoder_output, cross_attention_mask)
        x = x + self.dropout(encoder_decoder_output)  # Residual connection
        x = self.ln2(x)  # Layer normalization
        # Feed-Forward Network with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)  # Residual connection
        x = self.ln3(x)  # Layer normalization
        return x


# 4=========================
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

    def forward(self, hidden_state, k=None, v=None, attention_mask=None):
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
        if k is None:
            k = hidden_state
        if v is None:
            v = hidden_state
        key = self.k_linear(k)
        value = self.v_linear(v)
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
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (2, 8, 10, 64)


# 位置编码机制
class PositonalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositonalEncoding, self).__init__()
        # 初始化一个位置编码的矩阵, 维度是(max_len, d_model)
        self.pe = torch.zeros(max_len, d_model)
        self.base = 10000.0
        # Shape:(max_len, 1)
        # 创建一个从0到max_len-1的一维张量，数据类型为浮点数。
        # .unsqueeze(1)：在第二维（索引为1）添加一个维度，将形状从(max_len,)变为(max_len, 1)。
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Shape:(d_model//2,)
        # 创建一个从0到d_model-1的偶数索引序列，例如对于d_model=512，它会生成[0, 2, 4, ..., 510]。
        # -(math.log(self.base) / d_model)：计算一个缩放因子。
        # torch.exp(...)：对缩放后的序列进行指数运算，得到最终的除法项。
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(self.base) / d_model))
        # 根据公式填充位置编码矩阵
        # position * div_term：计算位置索引与除法项的乘积，形状为(max_len, d_model//2)。
        # orch.sin(position * div_term)：对乘积结果应用正弦函数，得到偶数索引的位置编码。, [:, 0::2]self.pe的所有行和偶数列
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        # print('self.pe', self.pe.shape)  # torch.Size([5000, 512])
        # .unsqueeze(0)：在第一维（索引为0）添加一个维度，将形状从(max_len, d_model)变为(1, max_len, d_model)。
        self.pe = self.pe.unsqueeze(0)  # Shape:(1, max_len)
        # print('self.pe', self.pe.shape)  # torch.Size([1, 5000, 512])

    def forward(self, x):
        # 获取输入张量的序列长度(batch_size, seq_len, embed_dim)
        seq_len = x.size(1)  # 10
        # print('self.pe[:, :seq_len]', self.pe[:, :seq_len].shape)  # torch.Size([1, 10, 512])
        return self.pe[:, :seq_len]

class MutiQueryAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MutiQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 64
        # 初始化Q,K,V投影矩阵
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, self.head_dim)
        self.v_linear = nn.Linear(hidden_size, self.head_dim)
        # 输出线性层
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]
        # print('hidden_state.shape', hidden_state.shape)  # torch.Size([2, 10, 512])
        seq_len = hidden_state.size()[1]
        # print('seq_len', seq_len)  # 10
        query = self.q_linear(hidden_state)
        # print('query.shape', query.shape)  # torch.Size([2, 10, 512])
        key = self.k_linear(hidden_state)
        # print('key.shape', key.shape)  # torch.Size([2, 10, 64])
        value = self.v_linear(hidden_state)
        query = self.split_head(query, self.num_heads)
        # print('query.shape', query.shape)  # torch.Size([2, 8, 10, 64])
        key = self.split_head(key, 1)
        value = self.split_head(value, 1)
        key = key.expand(-1, self.num_heads, -1, -1)
        # print(key.shape)  # torch.Size([2, 8, 10, 64])
        value = value.expand(-1, self.num_heads, -1, -1)
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        if attention_mask is not None:
            # 调整mask形状以匹配注意力分数
            attention_mask = attention_mask.unsqueeze(1)  # 添加head维度 -> [batch, 1, seq_len, seq_len]
            # print('attention_mask.shape', attention_mask.shape)  # torch.Size([2, 1, 10, 10])
            attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)  # 扩展为 [batch, num_heads, seq_len, seq_len]
            # print('attention_mask.shape', attention_mask.shape)  # torch.Size([2, 8, 10, 10])
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))  # 注意这里应该是负无穷
        # 对注意力分数进行归一化
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # print('attention_probs.shape', attention_probs.shape)  # torch.Size([2, 8, 10, 10])
        output = torch.matmul(attention_probs, value)
        # print('output.shape', output.shape)  # torch.Size([2, 8, 10, 64])
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)
        # print('output.shape', output.shape)  # torch.Size([2, 10, 512])
        output = self.o_linear(output)
        return output

    def split_head(self, x, head_num):
        batch_size = x.size()[0]
        return x.view(batch_size, -1, head_num, self.head_dim).transpose(1, 2)


# 将所有提供的类定义复制到这里
# (包括 TransformerEncoderBlock, EncoderDecoderAttention, TransformerDecoderBlock,
# PositonalEncoding, MultiHeadAttention, MutiQueryAttention)

def test_transformer_encoder_block():
    # 设置参数
    batch_size = 2
    seq_len = 10
    embed_dim = 512
    num_heads = 8
    ffn_hidden_dim = 2048
    dropout = 0.1

    # 创建模型
    encoder_block = TransformerEncoderBlock(embed_dim, num_heads, ffn_hidden_dim, dropout)

    # 创建随机输入
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 创建一个简单的mask（可选）
    # 修改掩码形状以适应多头注意力  多加一个1
    mask = torch.ones(batch_size, 1, seq_len, seq_len)  # # [batch_size, 1, seq_len, seq_len]

    # 前向传播
    output = encoder_block(x, mask)

    print("TransformerEncoderBlock 测试:")
    print(f"输入形状: {x.shape}")  # torch.Size([2, 10, 512])
    print(f"掩码形状: {mask.shape}")  # torch.Size([2, 1, 10, 10])
    print(f"输出形状: {output.shape}")  # torch.Size([2, 10, 512])
    print(f"输出形状是否正确: {output.shape == x.shape}")  # True
    print("-" * 50)


def test_transformer_decoder_block():
    # 设置参数
    batch_size = 2
    seq_len = 10
    embed_dim = 512
    num_heads = 8
    ffn_hidden_dim = 2048
    dropout = 0.1

    # 创建模型
    decoder_block = TransformerDecoderBlock(embed_dim, num_heads, ffn_hidden_dim, dropout)

    # 创建随机输入
    # 解码器的输入，形状为[batch_size, seq_len, embed_dim]，即[2, 10, 512]
    x = torch.randn(batch_size, seq_len, embed_dim)
    # 编码器的输出，形状与x相同，即[2, 10, 512], 这模拟了Transformer编码器处理源序列后的输出
    encoder_output = torch.randn(batch_size, seq_len, embed_dim)

    # 创建简单的masks（可选）
    # 自注意力掩码
    self_attention_mask = torch.ones(batch_size, seq_len, seq_len)
    # 交叉注意力掩码
    cross_attention_mask = torch.ones(batch_size, seq_len, seq_len)

    # 前向传播
    output = decoder_block(x, encoder_output, self_attention_mask, cross_attention_mask)

    print("TransformerDecoderBlock 测试:")
    print(f"输入形状: {x.shape}")  # torch.Size([2, 10, 512])
    print(f"编码器输出形状: {encoder_output.shape}")  # torch.Size([2, 10, 512])
    print(f"解码器输出形状: {output.shape}")  # torch.Size([2, 10, 512])
    print(f"输出形状是否正确: {output.shape == x.shape}")
    print("-" * 50)


def test_positional_encoding():
    # 设置参数
    batch_size = 2
    seq_len = 10
    embed_dim = 512

    # 创建模型
    pos_encoding = PositonalEncoding(embed_dim)

    # 创建随机输入
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 前向传播
    pe = pos_encoding(x)

    print("PositonalEncoding 测试:")
    print(f"输入形状: {x.shape}")  # torch.Size([2, 10, 512])
    print(f"位置编码形状: {pe.shape}")  # torch.Size([1, 10, 512])
    print(f"位置编码形状是否正确: {pe.shape == (1, seq_len, embed_dim)}")
    print("-" * 50)


def test_multi_head_attention():
    # 设置参数
    batch_size = 2
    seq_len = 10
    embed_dim = 512
    num_heads = 8

    # 创建模型
    mha = MultiHeadAttention(embed_dim, num_heads)

    # 创建随机输入
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 创建一个简单的mask（可选）
    mask = torch.ones(batch_size, seq_len, seq_len)

    # 前向传播
    output = mha(x, mask)

    print("MultiHeadAttention 测试:")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出形状是否正确: {output.shape == x.shape}")
    print("-" * 50)


def test_multi_query_attention():
    # 设置参数
    batch_size = 2
    seq_len = 10
    embed_dim = 512
    num_heads = 8

    # 创建模型
    mqa = MutiQueryAttention(embed_dim, num_heads)

    # 创建随机输入
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 创建一个简单的mask（可选）
    mask = torch.ones(batch_size, seq_len, seq_len)

    # 前向传播
    output = mqa(x, mask)

    print("MutiQueryAttention 测试:")
    print(f"输入形状: {x.shape}")  # torch.Size([2, 10, 512])
    print(f"输出形状: {output.shape}")  # torch.Size([2, 10, 512])
    print(f"输出形状是否正确: {output.shape == x.shape}")
    print("-" * 50)


if __name__ == "__main__":
    # test_transformer_encoder_block()
    # test_transformer_decoder_block()
    # test_positional_encoding()
    # test_multi_head_attention()
    test_multi_query_attention()
