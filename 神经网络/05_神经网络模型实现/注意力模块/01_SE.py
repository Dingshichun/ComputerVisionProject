# SE（Squeeze-and-Excitation）注意力机制通过显式建模通道间的依赖关系，对特征图的通道进行动态重校准。
# 核心流程分为三步：
# 1. ​​Squeeze​​：通过全局平均池化（GAP）压缩空间维度，生成通道描述符。
# 2. ​​Excitation​​：通过全连接层学习通道权重，使用 Sigmoid 激活生成 0-1 的权重系数。
# 3. ​​Reweight​​：将通道权重与原始特征图逐通道相乘，实现特征增强/抑制。

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()

        # Squeeze 操作：全局平均池化，是对每个通道进行全局平均池化
        # 比如输入 4 张 3 通道的大小为 224*224 的图像，即输入维度是 N*C*H*W=4*3*224*224
        # 那么经过全局平均池化后就变成 4*3*1*1，即每个通道被跟它尺寸一样大的卷积核卷积了，
        # 输出特征图大小为 1*1
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Excitation 操作：全连接层。为了捕获各通道之间的依赖性
        # 为了应用两个非线性激活函数，所以设置为两层，为了减少计算量设置了 reduction
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, _, _ = x.size() 
        # Squeeze 阶段：压缩空间维度
        y = self.gap(x).view(batch, channels)  # [B, C, 1, 1] → [B, C]
        # Excitation 阶段：学习通道权重
        y = self.fc(y).view(batch, channels, 1, 1)  # [B, C] → [B, C, 1, 1]
        # Reweight 阶段：应用权重
        return x * y.expand_as(x)


# 在 ResNet 的残差块中添加 SE 模块，即插即用。
# 通常放在卷积层之后，激活函数之前，或残差连接的分支路径上。
class ResBlockWithSE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)  # 在残差块中插入 SE 模块
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.se(x)  # 应用 SE 注意力
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual  # 残差连接
