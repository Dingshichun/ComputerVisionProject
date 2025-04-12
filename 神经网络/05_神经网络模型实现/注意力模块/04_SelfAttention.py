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


# 没看懂
class MultiHeadSelfAttention(nn.Module):
    """多头自注意力"""

    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size需可被heads整除"

        # 线性变换层
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]  # 批量大小
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 分割多头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # 生成Q/K/V
        Q = self.queries(queries)
        K = self.keys(keys)
        V = self.values(values)

        # 计算注意力分数
        energy = torch.einsum("nqhd,nkhd->nhqk", [Q, K])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = F.softmax(energy / (self.embed_size**0.5), dim=3)

        # 加权求和与合并多头
        out = torch.einsum("nhql,nlhd->nqhd", [attention, V])
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        return self.fc_out(out)


if __name__ == "__main__":
    # 1.将上面定义的 SelfAttention 类的功能逐步拆解
    # 输入形状为 (batch_size, seq_len, input_dim)=1*3*4，input_dim=4
    input = torch.randn(1, 3, 4)
    query_matrix = nn.Linear(
        in_features=4, out_features=4
    )  # 两层线性层，第一层必须和 input_dim 相同
    key_matrix = nn.Linear(in_features=4, out_features=4)
    value_matrix = nn.Linear(in_features=4, out_features=4)

    query = query_matrix(input)  # 查询
    key = key_matrix(input)  # 键
    value = value_matrix(input)  # 值
    scale = input.shape[-1] ** (-0.5)  # 缩放因子

    # 注意力分数。torch.matmul() 是矩阵相乘，遵循矩阵乘法
    # torch.mul() 是矩阵点积，即对应位置元素相乘
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    attention_weight = F.softmax(scores, dim=-1)  # 沿最后一个维度归一化
    context = torch.matmul(attention_weight, value)  # 加权求和
    print(context.shape)

    # 2.调用 SelfAttention 类
    # input = torch.randn(1, 3, 4)
    # module = SelfAttention(4)
    # out = module(input)
    # print(out.shape)
