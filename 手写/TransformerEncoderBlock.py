import torch
import torch.nn as nn
import torch.nn.functional as F
class TransformerEncoderBlock(nn.Module):
    def __init(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init()
        # Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        # Layer normalization
        self.n1 = nn.layerNorm(embed_dim)
        self.n2 = nn.layerNorm(embed_dim)
        # Feed-Forward Network (FNN)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.linear(ffn_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.droptout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        # Self-Attention with residual connection
        attention_output = self.self_attention(x, mask)
        x = x + self.droptout(attention_output) # residual
        x = self.ln1(x) # Layer normalization
        # Feed-Forward Network with residual connection
        ffn_output = self.fnn(x)
        x = x + self.droptout(ffn_output)  # residual connection
        x = self.ln2(x) # Layer normalization
        return x
