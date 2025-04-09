# 实现一个简化版的 Transformer 模型，
# 包含自注意力机制和前馈网络层，并移除了原版中一些复杂的设计
# 以便于理解和学习 Transformer 的基本原理

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

        # 通过线性变换得到 Q、K、V 矩阵
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size) 

    def forward(self, values, keys, query, mask):
        # 获取 batch size
        B = query.shape[0]

        # 拆分多头，并将数据重塑为 (B, seq_len, heads, head_dim)
        # 这里的 seq_len 是指输入序列的长度
        values = values.reshape(
            B, -1, self.heads, self.head_dim
        )  # (B, seq_len, heads, head_dim)
        keys = keys.reshape(B, -1, self.heads, self.head_dim)
        queries = query.reshape(B, -1, self.heads, self.head_dim)

        # 通过线性层变换
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 计算注意力分数
        energy = torch.einsum( 
            "bqhd,bkhd->bhqk", [queries, keys]
        )  # (B, heads, query_len, key_len)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # 应用注意力到 values 上
        out = torch.einsum("bhql,blhd->bqhd", [attention, values])
        out = out.reshape(B, -1, self.heads * self.head_dim)  # 合并多头

        # 最终线性层
        out = self.fc_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_dim):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_size, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_size)

    def forward(self, x):
        # 简单的前馈网络
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.ff = FeedForward(embed_size, ff_dim)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 残差连接 + 层归一化
        attention = self.attention(x, x, x, mask)
        x = self.norm1(attention + x)
        x = self.dropout(x)

        ff_out = self.ff(x)
        x = self.norm2(ff_out + x)
        x = self.dropout(x)
        return x


class SimplifiedTransformer(nn.Module):
    def __init__(
        self, vocab_size, embed_size, num_layers, heads, ff_dim, max_len, dropout=0.1
    ):
        super(SimplifiedTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_size))

        self.layers = nn.ModuleList(
            [
                EncoderLayer(embed_size, heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, seq_len = x.shape
        embeddings = self.embed(x)  # (B, seq_len, embed_size)
        embeddings += self.pos_embedding[:, :seq_len, :]

        out = self.dropout(embeddings)

        for layer in self.layers:
            out = layer(out, mask)

        return out


# 示例用法
if __name__ == "__main__":
    # 超参数
    vocab_size = 10000
    embed_size = 128
    num_layers = 2
    heads = 4
    ff_dim = 256
    max_len = 512
    dropout = 0.1

    # 初始化模型
    model = SimplifiedTransformer(
        vocab_size, embed_size, num_layers, heads, ff_dim, max_len, dropout
    )

    # 测试输入
    x = torch.randint(0, vocab_size, (1, 10))  # (batch_size, seq_len)
    out = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
