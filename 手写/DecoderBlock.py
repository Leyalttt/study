import torch
import torch.nn as nn
import torch.nn.functional as F
class EncoderDncoderBlock(nn.Module):
    def __init(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super(EncoderDncoderBlock, self).__init()
        # 这个Multi-HeadAttention forward的时候接受QKV对应的输入
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
    def forward(self, x, encoder_output, mask=None):
        # Calculate attention based on encoder's output(cross-attention)
        # KV:encoder_output，Q: X
        attention_output = self.cross_attention(x, encoder_output, encoder_output, mask)
        return attention_output

class TransformerDncoderBlock(nn.Module):
    def __init(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super(TransformerDncoderBlock, self).__init()
        # masked Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
