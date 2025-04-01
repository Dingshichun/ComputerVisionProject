"""
SSD目标检测与计数
by : DSC
date : 2025/04/01
"""

import cv2
import numpy as np

# 加载预训练模型，配置文件和类别标签，需要预先下载
model_config = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  # 模型配置文件
model_weights = "frozen_inference_graph.pb"  # 模型权重文件
class_names = "coco.names"  # COCO 数据集类别标签（共 90 类）

# 加载类别标签
with open(class_names, "r") as f:
    classes = f.read().strip().split("\n")

# 初始化 SSD 模型
net = cv2.dnn_DetectionModel(model_weights, model_config)
net.setInputSize(300, 300)  # 模型输入尺寸
net.setInputScale(1.0 / 127.5)  # 标准化参数（根据模型调整）
net.setInputMean((127.5, 127.5, 127.5))  # 均值归一化
net.setInputSwapRB(True)  # BGR 转 RGB

# 定义检测参数
conf_threshold = 0.5  # 置信度阈值
nms_threshold = 0.4  # 非极大值抑制阈值

# 读取视频流（或图像）
cap = cv2.VideoCapture(0)
counts = {}  # 统计各类别数量

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 执行检测
    class_ids, confidences, boxes = net.detect(frame, confThreshold=conf_threshold)

    # 应用非极大值抑制（NMS）
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # 重置计数
    counts = {}

    # 绘制检测框并统计
    if len(indices) > 0:
        for i in indices.flatten():
            class_id = class_ids[i]
            class_name = classes[class_id]
            counts[class_name] = counts.get(class_name, 0) + 1

            # 绘制检测框和标签
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{class_name}: {confidences[i]:.2f}"
            cv2.putText(
                frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

    # 显示统计结果
    y_offset = 30
    for cls, count in counts.items():
        cv2.putText(
            frame,
            f"{cls}: {count}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        y_offset += 30

    cv2.imshow("SSD Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
