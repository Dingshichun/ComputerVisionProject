'''
description:给图片添加滤镜
by:DSC
date:2025/03/31
'''
import cv2
import numpy as np

def apply_filter(image, filter_name):
    if filter_name == "grayscale":
        # 灰度化滤镜
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_name == "invert":
        # 反色（负片）滤镜
        return cv2.bitwise_not(image)
    elif filter_name == "vintage":
        # 怀旧滤镜（暖色调）
        rows, cols = image.shape[:2]
        vintage = np.zeros_like(image)
        # 红色通道增强
        vintage[:, :, 2] = np.clip(image[:, :, 2] * 0.7 + 40, 0, 255)
        # 绿色通道减弱
        vintage[:, :, 1] = np.clip(image[:, :, 1] * 0.5, 0, 255)
        # 蓝色通道减弱
        vintage[:, :, 0] = np.clip(image[:, :, 0] * 0.3, 0, 255)
        return vintage
    elif filter_name == "blur":
        # 高斯模糊
        return cv2.GaussianBlur(image, (15, 15), 0)
    elif filter_name == "edges":
        # Canny边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    else:
        return image

# 读取图片
image = cv2.imread("lena.jpg")

# 应用滤镜
filtered_grayscale = apply_filter(image, "grayscale")
filtered_invert = apply_filter(image, "invert")
filtered_vintage = apply_filter(image, "vintage")
filtered_blur = apply_filter(image, "blur")
filtered_edges = apply_filter(image, "edges")

# 显示结果（使用 Matplotlib）
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))

# 原始图片
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off") 

# 灰度化
plt.subplot(2, 3, 2)
plt.imshow(filtered_grayscale, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

# 反色
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(filtered_invert, cv2.COLOR_BGR2RGB))
plt.title("Invert")
plt.axis("off")

# 怀旧
plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(filtered_vintage, cv2.COLOR_BGR2RGB))
plt.title("Vintage")
plt.axis("off")

# 模糊
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(filtered_blur, cv2.COLOR_BGR2RGB))
plt.title("Blur")
plt.axis("off")

# 边缘检测
plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(filtered_edges, cv2.COLOR_BGR2RGB))
plt.title("Edges")
plt.axis("off")

plt.tight_layout() # 调整布局
plt.show()

