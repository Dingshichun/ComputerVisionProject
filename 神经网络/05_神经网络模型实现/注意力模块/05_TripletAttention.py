# Triplet Attention通过三个并行的分支捕获不同维度的跨维度交互关系，
# 在不降维的前提下融合通道与空间信息，显著提升特征表达能力。其核心创新在于：
# 1. ​​跨维度交互​​：通过张量旋转操作建立通道（C）与空间（H/W）的依赖关系；
# 2. ​​Z-Pool压缩​​：将通道维度压缩为2维（最大池化+平均池化），保留丰富特征的同时减少计算量；
# 3. ​​三支路结构​​：三个分支分别处理(C,H)、(C,W)和(H,W)维度的交互，最后通过平均聚合结果。

import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class TripletAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        # 分支1：通道 C 与空间 W 交互
        self.branch1 = nn.Sequential(
            BasicConv(2, 1, kernel_size, padding=(kernel_size - 1) // 2),
            BasicConv(1, 1, kernel_size, padding=(kernel_size - 1) // 2),
        )
        # ：通道 C 与空间 H 交互
        self.branch2 = nn.Sequential(
            BasicConv(2, 1, kernel_size, padding=(kernel_size - 1) // 2),
            BasicConv(1, 1, kernel_size, padding=(kernel_size - 1) // 2),
        )
        # 分支3：空间 H 与 W 交互
        self.branch3 = BasicConv(2, 1, kernel_size, padding=(kernel_size - 1) // 2)

    @staticmethod
    def channel_pool(x):
        return torch.cat(
            [torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim=1
        )

    def forward(self, x):
        # 分支1处理 (C, W)
        x1 = x.permute(0, 3, 1, 2)  # [B,W,H,C]->[B,C,H,W]
        x1 = self.channel_pool(x1)
        x1 = self.branch1(x1)
        x1 = x1.permute(0, 3, 1, 2)  # 恢复原始维度

        # 分支2处理 (C, H)
        x2 = x.permute(0, 2, 1, 3)  # [B,H,W,C]->[B,C,W,H]
        x2 = self.channel_pool(x2)
        x2 = self.branch2(x2)
        x2 = x2.permute(0, 3, 1, 2)

        # 分支3处理 (H, W)
        x3 = self.channel_pool(x)
        x3 = self.branch3(x3)

        # 聚合结果
        return (x1 + x2 + x3) / 3


# 在 ResNet 中插入Triplet Attention
class ResBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.attn = TripletAttention()  # 插入位置
        self.conv2 = nn.Conv2d(in_ch, in_ch, 3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.attn(x)  # 应用注意力
        x = self.conv2(x) + residual
