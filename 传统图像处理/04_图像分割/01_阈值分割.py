# 适合简单场景

import cv2
import numpy as np

# 读取图像并转为灰度
img = cv2.imread("lena.jpg", 0)

# 全局阈值分割
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 自适应阈值分割
adaptive_thresh = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# 显示结果
cv2.imshow("Global Threshold", thresh)
cv2.imshow("Adaptive Threshold", adaptive_thresh)
cv2.waitKey(0)
