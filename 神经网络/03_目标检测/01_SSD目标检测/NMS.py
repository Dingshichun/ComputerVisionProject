# 非极大值抑制的实现

import torch
import torchvision


def box_iou(box1, box2):
    # 计算交集坐标
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])

    # 计算交集和并集面积
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / union


def nms(
    predictions: torch.Tensor,
    conf_thres: float = 0.5,
    iou_thres: float = 0.45,
    max_det: int = 300,
    multi_class: bool = True,
) -> torch.Tensor:
    """
    非极大值抑制函数

    参数:
        predictions: 模型原始输出，形状为 [N, 6] (x1, y1, x2, y2, conf, cls)
        conf_thres: 置信度阈值，过滤低置信度预测框
        iou_thres: IoU阈值，用于抑制重叠框
        max_det: 最大保留检测框数量
        multi_class: 是否执行多类别NMS

    返回:
        筛选后的检测框 [M, 6] (x1, y1, x2, y2, conf, cls)
    """
    # 置信度过滤
    mask = predictions[:, 4] > conf_thres
    pred = predictions[mask]
    if pred.shape[0] == 0:
        return torch.zeros((0, 6), device=predictions.device)

    # 解析坐标、置信度、类别
    boxes = pred[:, :4]  # [x1, y1, x2, y2]
    scores = pred[:, 4]  # 置信度
    labels = pred[:, 5]  # 类别

    # 多类别处理
    if multi_class:
        unique_labels = labels.unique()
        keep = []
        for cls in unique_labels:
            cls_mask = labels == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            # 按置信度排序
            _, order = cls_scores.sort(descending=True)
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            # 执行NMS
            idx = torchvision.ops.nms(cls_boxes, cls_scores, iou_thres)
            keep.append(order[idx])

        keep = torch.cat(keep) if keep else torch.tensor([], dtype=torch.long)
    else:
        # 单类别直接处理
        _, order = scores.sort(descending=True)
        boxes = boxes[order]
        scores = scores[order]
        keep = torchvision.ops.nms(boxes, scores, iou_thres)

    # 截断至最大检测数
    return pred[keep[:max_det]]
