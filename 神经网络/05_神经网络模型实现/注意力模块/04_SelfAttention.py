# 自注意力机制（self-Attention）
# 核心思想是通过计算输入序列中每个位置的全局上下文权重，动态关注重要特征。

# 自注意力通常由​​查询（Query）​​、​​键（Key）​​和​​值（Value）​​三个部分组成，通过以下步骤实现：
# 1. ​​计算注意力分数​​：通过查询与键的相似度生成权重。
# 2. ​​Softmax 归一化​​：将分数转换为概率分布。
# 3. ​​加权求和​​：根据权重对值进行加权融合。

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """单头自注意力"""

    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.scale = input_dim ** (-0.5)  # 缩放因子，防止点积过大

    def forward(self, x):
        """
        x: 输入张量，形状为 (batch_size, seq_len, input_dim)
        batch_size 是批次大小，seq_len 是序列长度，input_dim 是词嵌入的维度
        比如将一个单词进行 64 维词嵌入，也就是用一个 64 维向量表示，那么 input_dim=64,
        训练时一组输入 100 个单词，那么 seq_len=100。batch_size 就是输入的组数
        """
        batch_size, seq_len, _ = x.size()

        # 生成 Query、Key、Value
        query = self.query_proj(x)  # (batch_size, seq_len, input_dim)
        key = self.key_proj(x)  # (batch_size, seq_len, input_dim)
        value = self.value_proj(x)  # (batch_size, seq_len, input_dim)

        # 计算注意力分数
        scores = (
            torch.matmul(query, key.transpose(-2, -1)) * self.scale
        )  # (batch_size, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)  # 沿最后一个维度归一化

        # 加权求和
        context = torch.matmul(attn_weights, value)  # (batch_size, seq_len, input_dim)
        return context


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding维度必须能被头数整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性变换层（Q/K/V共享同一输入）
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # 输入形状: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim)
        )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # 步骤1：生成Q/K/V（自注意力共用输入）
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # 分割为Q/K/V

        # 步骤2：分割多头
        Q = self.split_heads(q)  # [batch, heads, seq_len_q, head_dim]
        K = self.split_heads(k)
        V = self.split_heads(v)

        # 步骤3：计算注意力并加权聚合
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 步骤4：合并多头
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # 步骤5：输出投影
        output = self.output_proj(attn_output)
        return output, attn_weights


if __name__ == "__main__":

    # # 一、单头自注意力

    # # 1.将上面定义的 SelfAttention 类的功能逐步拆解
    # # 输入形状为 (batch_size, seq_len, input_dim)=1*3*4，input_dim=4
    # input = torch.randn(1, 3, 4)
    # query_operate = nn.Linear(
    #     in_features=4, out_features=4
    # )  # 两层线性层，第一层必须和 input_dim 相同
    # key_operate = nn.Linear(in_features=4, out_features=4)
    # value_operate = nn.Linear(in_features=4, out_features=4)

    # query = query_operate(input)  # 查询
    # key = key_operate(input)  # 键
    # value = value_operate(input)  # 值
    # scale = input.shape[-1] ** (-0.5)  # 缩放因子

    # # 注意力分数。torch.matmul() 是矩阵相乘，遵循矩阵乘法
    # # torch.mul() 是矩阵点积，即对应位置元素相乘
    # scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # attention_weight = F.softmax(scores, dim=-1)  # 沿最后一个维度归一化
    # context = torch.matmul(attention_weight, value)  # 加权求和
    # print(context.shape)

    # 2.调用 SelfAttention 类
    # input = torch.randn(1, 3, 4)
    # module = SelfAttention(4)
    # out = module(input)
    # print(out.shape)

    multiHead = MultiHeadSelfAttention(16, 4)
    input1 = torch.randn(1, 3, 16)
    out = multiHead(input1)
    print(out[0])
