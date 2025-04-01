'''
图像目标识别与计数
by:DSC
date:2025/04/01
'''
import cv2
import numpy as np

# 读取图像
image = cv2.imread('red.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # 转换为 HSV 颜色空间

# 设定红色阈值范围（HSV）
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

# 生成红色掩膜
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2) 

# 形态学操作去噪
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭合操作填充空洞，闭操作先膨胀后腐蚀
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 开放操作去除噪声，开操作先腐蚀后膨胀

# 查找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 过滤小轮廓并计数
min_area = 500
count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > min_area:
        count += 1
        # 绘制轮廓
        cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

# 显示计数结果
cv2.putText(image, f'Count: {count}', (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()