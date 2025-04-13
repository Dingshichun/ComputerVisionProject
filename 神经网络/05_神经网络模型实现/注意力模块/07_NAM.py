# NAM（Normalization-based Attention Module）是一种轻量级注意力机制，
# 通过​​批归一化（BatchNorm）的缩放因子​​直接衡量通道/空间维度的重要性，
# 无需额外参数（如 SE 中的全连接层或 CBAM 中的卷积层），
#
# 其核心创新点包括：
# 1. ​​通道注意力​​：利用 BN 层的缩放因子（scale factor）表示通道方差，权重计算方式为：
# weight_bn = BN.weight.abs() / sum(BN.weight.abs())
# 通过归一化后的权重对特征图进行通道加权。
# 2. ​​空间注意力​​：将 BN 的缩放因子应用于空间维度，
# 通过​​像素归一化​​（Pixel Normalization）衡量空间位置的重要性。
# 3. ​​残差连接​​：将与原始输入通过 Sigmoid 激活后相乘，保留原始特征信息。

import torch
import torch.nn as nn


class NAM(nn.Module):
    def __init__(self, channels):
        super(NAM, self).__init__()
        # 通道注意力
        self.bn_ch = nn.BatchNorm2d(channels, affine=True)
        # 空间注意力（定义空间维度的 BN）
        self.bn_sp = nn.BatchNorm2d(1, affine=True)

    def channel_att(self, x):
        residual = x
        x = self.bn_ch(x)
        # 计算通道权重
        channel_weights = self.bn_ch.weight.data.abs() / torch.sum(
            self.bn_ch.weight.data.abs()
        )
        x = x.permute(0, 2, 3, 1) * channel_weights
        x = x.permute(0, 3, 1, 2)
        return torch.sigmoid(x) * residual

    def spatial_att(self, x):
        residual = x
        # 沿通道维度求均值，降维至 1 通道
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.bn_sp(x)
        # 计算空间权重
        spatial_weights = self.bn_sp.weight.data.abs() / torch.sum(
            self.bn_sp.weight.data.abs()
        )
        x = x * spatial_weights.view(1, -1, 1, 1)
        return torch.sigmoid(x) * residual

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# 在 ResNet 的残差块中嵌入 NAM
class ResBlock_NAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.nam = NAM(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.nam(x)  # 嵌入 NAM 模块
        x = self.conv2(x)
        return x + residual
