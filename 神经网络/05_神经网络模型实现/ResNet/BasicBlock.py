# ResNet的核心是​​残差块（Residual Block）​​，分为两种类型:BasicBlock 和 Bottleneck，
# BasicBlock ​​包含两个 3×3 卷积层，适用于较浅的网络结构。
# Bottleneck 通过 1×1 卷积降维/升维，减少计算量，适合较深网络

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    # expansion 是扩展因子，确保输入残差块的图像，和经过残差块处理后的输出通道数相同，
    # 能够进行相加操作
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        # 如果 stride != 1 ，那么经过卷积之后，图像的尺寸和和原图不一样，
        # 所以原图和经过卷积的图就不能通过捷径分支直接相加
        # 同样的，in_channels != out_channels * self.expansion ，这里定义 expansion=1
        # 表示输入通道数和经过残差块之后的输出通道数不相同，也不能进行直接相加操作。
        # 此时捷径分支上的操作就是保证原图像经过捷径分支处理后，
        # 和原图像经过 BasicBlock 块处理后的维度相同，可以直接相加。
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class BottleNeck(nn.Module):
    # expansion 是扩展因子，确保输入残差块的图像，和经过残差块处理后的输出通道数相同，
    # 能够进行相加操作。Bottleneck 残差块会将输入的图像通道数减少到原来的四分之一，
    # 所以最后要乘以 4 进行输出，才能和捷径分支上的原输入图像进行相加操作。
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.shortcut = nn.Sequential()

        # 这里和上面 BasicBlock 一样
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)
