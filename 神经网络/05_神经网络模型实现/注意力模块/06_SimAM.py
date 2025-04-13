# SimAM（Simple, Parameter-Free Attention Module）的灵感来源于神经科学理论，
# 认为信息丰富的神经元会抑制周围神经元（空域抑制效应）。
# 其通过能量函数衡量神经元的重要性：
# 能量越低​​的神经元越重要，注意力权重通过归一化能量值得到

# 关键实现细节​​
# 1. ​​无参数设计​​：
# 所有计算均基于输入特征图的统计量（方差），无需可学习的权重矩阵。
# 2. ​​高效计算​​：
# 通过逐元素操作和广播机制实现，计算复杂度为 O(C*H*W)，适合嵌入到深层网络。
# 3. ​​嵌入方式​​：
# 可作为即插即用模块，添加到卷积层后。

import torch
import torch.nn as nn


class SimAM(nn.Module):
    def __init__(self, lambda_val=1e-4):
        super(SimAM, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        # 输入x形状: (B, C, H, W)
        B, C, H, W = x.size()

        # 计算通道维度上的均值和方差
        mu = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        sigma = x.var(dim=1, keepdim=True)  # (B, 1, H, W)

        # 计算能量值（逐元素操作）
        energy = (x - mu).pow(2) / (sigma + self.lambda_val)  # (B, C, H, W)
        energy = 1.0 / (1.0 + energy)  # 能量越低，权重越高

        # 生成注意力权重（无需 Sigmoid，直接作为缩放因子）
        return energy.expand_as(x)


# 即插即用的嵌入示例
class EmbedSimAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.simam = SimAM()

    def forward(self, x):
        x = self.conv(x)
        x = self.simam(x)
        return x
