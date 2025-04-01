# 适合简单场景

import cv2

# 读取图像并转为灰度
img = cv2.imread("lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny边缘检测
edges = cv2.Canny(gray, 100, 200)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# 显示结果
cv2.imshow("Edges", edges)
cv2.imshow("Contours", result)
cv2.waitKey(0)
