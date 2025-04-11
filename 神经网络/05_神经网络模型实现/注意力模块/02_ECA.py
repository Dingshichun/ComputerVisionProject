# ECA (Efficient Channel Attention)
# 核心是通过一维卷积替代传统通道注意力中的全连接层，实现轻量级通道交互。
# 参数量较少，仅 k_size，一般是 3~9 个

import torch
from torch import nn


class ECA_Block(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,  # 保持输出维度不变
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化 [b,c,h,w] -> [b,c,1,1]
        y = self.avg_pool(x)
        # 维度变换 [b,c,1,1] -> [b,1,c] -> 一维卷积 -> [b,c,1]
        y = (
            self.conv(y.squeeze(-1).transpose(-1, -2))  # 移除最后一个维度并转置
            .transpose(-1, -2)
            .unsqueeze(-1)
        )  # 恢复维度
        # 生成注意力权重并扩展维度
        y = self.sigmoid(y)
        return x * y.expand_as(x)  # 广播到原特征图尺寸


# 在 CNN 中插入 ECA 模块
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.eca = ECA_Block(64)  # 输入通道需与前一层的输出一致
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.eca(x)  # 在卷积层后插入
        x = self.conv2(x)
        return x
