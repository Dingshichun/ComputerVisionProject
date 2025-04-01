# 图像配准是计算机视觉中的一个重要任务，通常用于将两幅图像对齐，以便进行比较或融合。
# 在本示例中，我们将使用OpenCV库中的SIFT算法来实现图像配准。
# SIFT（尺度不变特征变换）是一种用于检测和描述局部特征的算法，具有旋转、缩放和光照不变性。
# 该算法在图像配准中非常有效，尤其是在处理具有不同视角或光照条件的图像时。

import cv2
import numpy as np

# 读取图像（灰度图）
ref_img = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
align_img = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 检测关键点与描述符
kp1, des1 = sift.detectAndCompute(ref_img, None)
kp2, des2 = sift.detectAndCompute(align_img, None)

# FLANN匹配器参数
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 特征匹配
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's比率测试筛选匹配点
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

# 计算单应性矩阵
MIN_MATCH_COUNT = 4
if len(good) >= MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 应用变换
    h, w = ref_img.shape
    aligned_img = cv2.warpPerspective(align_img, H, (w, h))
    
    # 显示结果
    cv2.imshow('Aligned Image', aligned_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"匹配点不足，仅找到 {len(good)}/{MIN_MATCH_COUNT} 个匹配点。")