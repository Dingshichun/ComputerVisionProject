# 颜色阈值速度快，适合纯色背景，但可能误分割相似颜色区域。

import cv2
import numpy as np


def replace_background_color(img_path, new_bg_color=(255, 255, 255)):
    # 读取图像
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义蓝色背景的HSV范围（可根据实际颜色调整）
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # 生成掩膜
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 形态学处理：去噪
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算填补空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算消除噪声

    # 反转掩膜（背景为白色，前景为黑色）
    mask_inv = cv2.bitwise_not(mask)

    # 提取前景
    foreground = cv2.bitwise_and(img, img, mask=mask_inv)

    # 创建新背景（纯色）
    new_bg = np.full_like(img, new_bg_color)
    new_bg = cv2.bitwise_and(new_bg, new_bg, mask=mask)

    # 合成图像
    result = cv2.add(foreground, new_bg)
    return result


# 使用示例
result = replace_background_color(
    "image.jpg", new_bg_color=(0, 255, 0)
)  # 替换为绿色背景
cv2.imwrite("result.jpg", result)
