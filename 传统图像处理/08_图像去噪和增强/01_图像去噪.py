import cv2
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('lena.jpg')

# 去噪方法
denoised_median = cv2.medianBlur(img, 5)
denoised_gaussian = cv2.GaussianBlur(img, (5,5), 0)
denoised_bilateral = cv2.bilateralFilter(img, 9, 75, 75)
denoised_nlmeans = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# 将 BGR 转换为 RGB（供 matplotlib 显示）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
denoised_median_rgb = cv2.cvtColor(denoised_median, cv2.COLOR_BGR2RGB)
denoised_gaussian_rgb = cv2.cvtColor(denoised_gaussian, cv2.COLOR_BGR2RGB)
denoised_bilateral_rgb = cv2.cvtColor(denoised_bilateral, cv2.COLOR_BGR2RGB)
denoised_nlmeans_rgb = cv2.cvtColor(denoised_nlmeans, cv2.COLOR_BGR2RGB)

# 创建子图布局
plt.figure(figsize=(15, 8))  # 调整画布大小

# 显示原始图像
plt.subplot(2, 3, 1)  # 2行3列，第1个位置
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# 显示中值滤波结果
plt.subplot(2, 3, 2)  # 2行3列，第2个位置
plt.imshow(denoised_median_rgb)
plt.title('Median Filter')
plt.axis('off')

# 显示高斯滤波结果
plt.subplot(2, 3, 3)  # 2行3列，第3个位置
plt.imshow(denoised_gaussian_rgb)
plt.title('Gaussian Filter')
plt.axis('off')

# 显示双边滤波结果
plt.subplot(2, 3, 4)  # 2行3列，第4个位置
plt.imshow(denoised_bilateral_rgb)
plt.title('Bilateral Filter')
plt.axis('off')

# 显示非局部均值结果
plt.subplot(2, 3, 5)  # 2行3列，第5个位置
plt.imshow(denoised_nlmeans_rgb)
plt.title('NL-Means Filter')
plt.axis('off')

# 调整子图间距
plt.subplots_adjust(wspace=0.1, hspace=0.3)  # 控制水平和垂直间距
plt.show()
