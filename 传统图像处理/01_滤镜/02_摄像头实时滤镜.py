"""
description:摄像头添加滤镜
by:DSC
date:2025/03/31
"""

import cv2
import numpy as np


# 定义所有滤镜函数
def grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def invert(frame):
    '''反色滤镜'''
    return cv2.bitwise_not(frame)


def vintage(frame):
    # 暖色调怀旧效果
    vintage_img = frame.copy()
    vintage_img[:, :, 0] = np.clip(frame[:, :, 0] * 0.3, 0, 255)  # 减弱蓝色
    vintage_img[:, :, 1] = np.clip(frame[:, :, 1] * 0.5, 0, 255)  # 减弱绿色
    vintage_img[:, :, 2] = np.clip(frame[:, :, 2] * 0.7 + 40, 0, 255)  # 增强红色
    return vintage_img


def edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # 转为3通道显示


def blur(frame):
    '''高斯模糊滤镜'''
    return cv2.GaussianBlur(frame, (15, 15), 0)


def sharpen(frame):
    '''锐化滤镜'''
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)


def sepia(frame):
    # 棕褐色滤镜
    sepia_filter = np.array(
        [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
    )
    return cv2.transform(frame, sepia_filter)


# 滤镜映射表（按键: 滤镜函数）
filters = {
    ord("1"): ("Original", lambda x: x),  # 原图
    ord("2"): ("Grayscale", grayscale),
    ord("3"): ("Invert", invert),
    ord("4"): ("Vintage", vintage),
    ord("5"): ("Edges", edges),
    ord("6"): ("Blur", blur),
    ord("7"): ("Sharpen", sharpen),
    ord("8"): ("Sepia", sepia),
}

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
current_filter = filters[ord("1")]  # 默认使用原图

while True:
    ret, frame = cap.read() # 读取摄像头画面
    if not ret: # 检测摄像头是否正常工作
        break

    # 应用当前滤镜
    filter_name, filter_func = current_filter # 获取当前滤镜名称和函数
    filtered_frame = filter_func(frame) # 应用滤镜函数

    # 若滤镜输出是灰度图，转为 BGR 显示
    if len(filtered_frame.shape) == 2:
        filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)

    # 在画面左上角显示当前滤镜名称
    cv2.putText(
        filtered_frame,
        f"Filter: {filter_name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    # 显示画面
    cv2.imshow("Real-time Filter", filtered_frame)

    # 检测按键输入
    key = cv2.waitKey(1) & 0xFF
    if key in filters:
        current_filter = filters[key]  # 切换滤镜
    elif key == ord("q"):  # 按 q 退出
        break

cap.release() # 释放摄像头
cv2.destroyAllWindows() # 关闭所有窗口
