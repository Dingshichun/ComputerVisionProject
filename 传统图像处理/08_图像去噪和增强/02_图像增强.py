import cv2
import numpy as np

def enhance_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: 图像无法读取")
        return

    # 调整对比度和亮度
    alpha = 1.2  # 对比度系数 (1.0 无变化)
    beta = 15    # 亮度增量 (0 无变化)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 伽马校正
    gamma = 0.7  # 伽马值 (1.0 无变化)
    gamma_corrected = np.power(adjusted / 255.0, gamma) * 255
    gamma_corrected = gamma_corrected.astype(np.uint8)

    # YUV颜色空间CLAHE增强
    yuv = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)
    yuv_clahe = cv2.merge([y_clahe, u, v])
    clahe_image = cv2.cvtColor(yuv_clahe, cv2.COLOR_YUV2BGR)

    # 双边滤波降噪
    denoised = cv2.bilateralFilter(clahe_image, d=9, sigmaColor=75, sigmaSpace=75)

    # 锐化处理
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

    # 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Enhanced Image', sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用示例
enhance_image('lena.jpg')