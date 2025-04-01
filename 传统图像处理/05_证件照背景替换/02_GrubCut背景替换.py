# GrabCut：分割更精确，适合复杂边缘（如头发），但需要合理初始化区域。

import cv2
import numpy as np


def replace_background_grabcut(img_path, new_bg_color=(255, 255, 255)):
    img = cv2.imread(img_path)
    mask = np.zeros(img.shape[:2], np.uint8)

    # 定义GrabCut的初始矩形区域（假设人物在中央）
    h, w = img.shape[:2]
    rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))  # (x, y, w, h)

    # 初始化GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # 生成掩膜（0-背景，1-前景，2-可能背景，3-可能前景）
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # 提取前景
    foreground = img * mask[:, :, np.newaxis]

    # 创建新背景
    new_bg = np.full_like(img, new_bg_color)
    new_bg = new_bg * (1 - mask[:, :, np.newaxis])

    # 合成图像
    result = foreground + new_bg
    return result


# 使用示例
result = replace_background_grabcut(
    "image.jpg", new_bg_color=(255, 0, 0)
)  # 替换为红色背景
cv2.imwrite("result_grabcut.jpg", result)
