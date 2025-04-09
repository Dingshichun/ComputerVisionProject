# 该代码用于加载训练好的 SSD 模型，并对输入图像进行目标检测

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from ssd import SSD
from NMS import nms


def detect(image_path, model_path="ssd.pth", conf_threshold=0.5):
    # 加载模型
    model = SSD(num_classes=21)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 图像预处理
    image = Image.open(image_path).convert("RGB")
    image = F.to_tensor(image).unsqueeze(0)

    # 推理
    with torch.no_grad():
        pred_loc, pred_conf = model(image)

    # 解码预测框
    boxes, scores = decode_predictions(pred_loc, pred_conf, conf_threshold)
    return boxes, scores


def decode_predictions(loc, conf, conf_threshold):
    # 简化解码逻辑（需结合先验框生成）
    boxes = loc[0]  # 实际需结合先验框计算
    scores = F.softmax(conf[0], dim=1)
    # NMS后处理
    keep = nms(boxes, scores[:, 1:], iou_threshold=0.5)
    return boxes[keep], scores[keep]
