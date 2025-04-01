# 分水岭算法是一种图像分割技术，常用于分离重叠的物体
# 该算法将图像视为地形图，水流从高处流向低处，形成分水岭线

import cv2
import numpy as np

# 读取图像
img = cv2.imread("lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化预处理
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 去噪
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 确定背景区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 确定前景区域（距离变换）
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# 计算未知区域
unknown = cv2.subtract(sure_bg, sure_fg)

# 标记连通区域
ret, markers = cv2.connectedComponents(sure_fg)
markers += 1
markers[unknown == 255] = 0

# 应用分水岭算法
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]  # 标记边界为红色

# 显示结果
cv2.imshow("Watershed", img)
cv2.waitKey(0)
