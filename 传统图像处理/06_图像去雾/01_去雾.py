'''
1. ​计算暗通道：对图像的每个像素取 RGB 三通道的最小值，并在局部窗口中取最小值。
2. ​估计大气光(A) ​:取暗通道中最亮的像素对应的原始图像区域的 RGB 均值。
3. ​估计透射率（t）​：通过暗通道和大气光计算透射率。
4. ​优化透射率：使用引导滤波（Guided Filter）细化透射率图。
5. ​恢复无雾图像：根据大气散射模型重建清晰图像。
'''
import cv2
import numpy as np

def dark_channel(img, window_size=15):
    # 计算暗通道（确保输入是float32）
    min_rgb = cv2.min(img[:, :, 0], cv2.min(img[:, :, 1], img[:, :, 2]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark = cv2.erode(min_rgb, kernel)
    return dark

def estimate_atmospheric_light(img, dark, percentile=0.1):
    # 估计大气光（输入图像需为float32）
    h, w = img.shape[:2]
    num_pixels = h * w
    num_top = int(num_pixels * percentile / 100)
    indices = np.argsort(dark.ravel())[-num_top:]
    atmospheric_light = np.mean(img.reshape(-1, 3)[indices], axis=0)
    return atmospheric_light

def estimate_transmission(img, atmospheric_light, omega=0.95, window_size=15):
    # 估计透射率（确保归一化为float32）
    normalized_img = img.astype(np.float32) / atmospheric_light.astype(np.float32)
    transmission = 1 - omega * dark_channel(normalized_img, window_size)
    return transmission

def guided_filter(img, guide, radius=60, eps=1e-6):
    # 引导滤波（强制转换为float32）
    img = img.astype(np.float32)
    guide = guide.astype(np.float32)
    
    mean_I = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
    mean_p = cv2.boxFilter(img, cv2.CV_32F, (radius, radius))
    mean_Ip = cv2.boxFilter(guide * img, cv2.CV_32F, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
    
    refined_transmission = mean_a * guide + mean_b
    return refined_transmission

def dehaze(img, window_size=15, omega=0.95, radius=40, eps=1e-6, t0=0.1):
    # 主函数（统一使用float32）
    img = img.astype(np.float32) / 255.0  # 改为float32
    dark = dark_channel(img)
    atmospheric_light = estimate_atmospheric_light(img, dark)
    
    # 计算透射率并优化
    transmission = estimate_transmission(img, atmospheric_light, omega, window_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)  # 引导图转为float32
    transmission = guided_filter(transmission, gray, radius, eps)
    transmission = np.clip(transmission, t0, 1.0)
    
    # 恢复无雾图像
    result = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result

# 测试
img = cv2.imread('haze3.jpg')
result = dehaze(img)
cv2.imshow('Dehazed', result)
cv2.waitKey(0)
cv2.destroyAllWindows()