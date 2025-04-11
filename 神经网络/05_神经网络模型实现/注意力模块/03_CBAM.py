# CBAM（Convolutional Block Attention Module）
# CBAM 由通道注意力（Channel Attention）和空间注意力（Spatial Attention）
# 两个子模块级联组成，通过自适应特征增强提升模型性能

# 1.通道注意力模块（Channel Attention Module）
# 通过全局平均池化和最大池化捕获通道维度的全局信息，结合共享的 MLP 生成通道权重

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super().__init__()

        # 压缩输入图像的空间维度，保留通道维度。包括平均池化和最大池化
        # 比如输入图像 Channel*Height*Width=3*224*224，那么输出就是 3*1*1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的压缩-恢复模块（可用全连接或卷积实现）
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))  # 平均池化分支
        max_out = self.shared_mlp(self.max_pool(x))  # 最大池化分支
        channel_weights = self.sigmoid(
            avg_out + max_out
        )  # 融合权重，得到通道注意力权重

        return x * channel_weights  # 通道注意力权重应用到原始输入图像。


# 2.空间注意力模块（Spatial Attention Module）
# 空间注意力通过通道维度的池化获取空间显著性，结合卷积生成空间权重


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2  # 保持特征图尺寸不变
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 因为通常都是输入多幅图像（N*Channel*Height*Width）
        # 所以 Channel 所在的 dim 是 1，torch.mean() 和 torch.max()
        # 中的参数 dim=1 ，表示求通道维度的均值和最大值
        # 比如输入维度 4*3*224*224,3 表示通道数，处理后维度变为 4*1*224*224
        # 即通道维度被压缩了，
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道最大池化

        # 经过上面压缩通道维度的操作之后，得到两个单通道的图像，
        # 在通道维度进行拼接之后就得到双通道的图像
        spatial_features = torch.cat([avg_out, max_out], dim=1)  # 拼接特征

        # 双通道的图像卷积之后得到单通道图像，再经过 Sigmoid 激活得到空间注意力权重
        spatial_weights = self.sigmoid(
            self.conv(spatial_features)
        )  # 生成空间注意力权重

        return x * spatial_weights  # 空间注意力权重应用到原始输入图像


# 3.CBAM 模块
class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)  # 先通道注意力
        x = self.spatial_att(x)  # 再空间注意力
        return x


# 4.插入 ResNet 的残差块中


class SEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, padding=1)
        self.cbam = CBAM(planes)  # 在残差连接前插入 CBAM
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.cbam(out)  # 应用 CBAM
        out = self.conv2(out)
        out += residual
        return out


if __name__ == "__main__":
    # 因为上面 ChannelAttention 中的 reduction_ratio=16 ，
    # 所以通道的位置最少为 16 才能确保维度匹配
    input = torch.randn(4, 16, 224, 224)
    cam = ChannelAttention(16)
    sam = SpatialAttention()
    cbam = CBAM(16)
    cam_output = cam(input)
    sam_output = sam(input)
    cbam_output = cbam(input)
    print(f"input'shape is:{input.shape}")
    print(f"cam_output'shape is:{cam_output.shape}")
    print(f"sam_output'shape is:{sam_output.shape}")
    print(f"cbam_output'shape is:{cbam_output.shape}")
