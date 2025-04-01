# k-means聚类分割适合颜色区分明显的图像

import cv2
import numpy as np

# 读取图像
img = cv2.imread("lena.jpg")
data = img.reshape((-1, 3)).astype(np.float32)

# 定义 K-Means 参数
K = 3  # 聚类数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 重建图像
center = np.uint8(center)
result = center[label.flatten()]
result = result.reshape(img.shape)

# 显示结果
cv2.imshow("K-Means", result)
cv2.waitKey(0)
