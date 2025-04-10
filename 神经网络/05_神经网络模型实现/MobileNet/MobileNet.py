# mobilenet 最核心的是深度可分离卷积（Depthwise Separable Convolution）
# 就是将传统的卷积分为深度卷积​​（Depthwise Convolution）和​​逐点卷积​​（Pointwise Convolution）
# 深度卷积​​：比如每个输入通道单独应用一个 3×3 卷积核，生成与输入通道数相同的特征图。
# 逐点卷积​​：通过 1×1 卷积调整通道数，实现跨通道的特征融合。

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        # 逐点卷积
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)  # 使用ReLU6增强低精度稳定性

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


# MobileNet 由​​标准卷积层​​和​​堆叠的深度可分离卷积块​​构成
# 结构特点：
# 1.首层为标准卷积​​：输入尺寸 224×224 时，首层使用 3×3 卷积（stride=2）降采样至 112×112。
# ​2.​堆叠深度可分离卷积块​​：按配置文件 cfg 定义通道数和步长，交替使用 stride=1 和 stride=2进行特征提取。
# ​3.​全局平均池化与分类层​​：最终输出 1024 维特征向量，通过全连接层映射到目标类别数


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        # 动态调整通道数（宽度乘数）
        input_channels = int(32 * width_mult)
        last_channels = int(1024 * width_mult)

        self.features = nn.Sequential(
            # 首层标准卷积
            nn.Conv2d(
                3, input_channels, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True),
            # 堆叠深度可分离卷积块
            DepthwiseSeparableConv(input_channels, int(64 * width_mult)),
            DepthwiseSeparableConv(
                int(64 * width_mult), int(128 * width_mult), stride=2
            ),
            DepthwiseSeparableConv(int(128 * width_mult), int(128 * width_mult)),
            DepthwiseSeparableConv(
                int(128 * width_mult), int(256 * width_mult), stride=2
            ),
            DepthwiseSeparableConv(int(256 * width_mult), int(256 * width_mult)),
            DepthwiseSeparableConv(
                int(256 * width_mult), int(512 * width_mult), stride=2
            ),
            # 重复多个块
            *[
                DepthwiseSeparableConv(int(512 * width_mult), int(512 * width_mult))
                for _ in range(5)
            ],
            DepthwiseSeparableConv(int(512 * width_mult), last_channels, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        )
        self.classifier = nn.Linear(last_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = MobileNet(width_mult=0.5)  # 轻量化版本
    # print(model)
    tensor=torch.randn(1,3,224,224) # 输入数据为 N*C*W*H，表示批次*通道数*宽*高
    output=model(tensor)
    print(output.shape)
