# 定义 SSD 模型
# 该模型使用 VGG16 作为骨干网络，添加额外卷积层和多尺度预测头

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class SSD(nn.Module):
    def __init__(self, num_classes=21, backbone="vgg16"):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.backbone = self._build_backbone(backbone)
        self.extras = self._build_extra_layers()
        self.multibox = self._build_multibox()

    def _build_backbone(self, backbone):
        # 修改VGG16作为骨干网络
        base = vgg16(pretrained=True).features
        conv4_3_idx = 23  # VGG16的Conv4_3层
        layers = list(base.children())[: conv4_3_idx + 1]
        # 调整池化层参数
        layers[-1] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        return nn.Sequential(*layers)

    def _build_extra_layers(self):
        # 添加额外卷积层生成多尺度特征图
        extras = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        return extras

    def _build_multibox(self):
        # 多尺度预测头
        loc_layers = []
        conf_layers = []
        # 不同特征图的默认框数量配置
        num_anchors = [4, 6, 6, 6, 4, 4]
        in_channels = [512, 1024, 512, 256, 256, 256]
        for i, (n, c) in enumerate(zip(num_anchors, in_channels)):
            loc_layers += [nn.Conv2d(c, n * 4, kernel_size=3, padding=1)]
            conf_layers += [
                nn.Conv2d(c, n * self.num_classes, kernel_size=3, padding=1)
            ]
        return (nn.ModuleList(loc_layers), nn.ModuleList(conf_layers))

    def forward(self, x):
        # 前向传播生成多尺度特征
        sources, loc, conf = [], [], []
        x = self.backbone(x)
        sources.append(x)
        x = self.extras(x)
        sources.append(x)

        # 对每个特征图进行预测
        for x, l, c in zip(sources, self.multibox[0], self.multibox[1]):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # 拼接所有预测结果
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return loc, conf
