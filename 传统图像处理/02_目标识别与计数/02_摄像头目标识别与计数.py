"""
摄像头目标识别与计数
by:DSC
date:2025/04/01
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # 使用摄像头捕获视频流
fgbg = cv2.createBackgroundSubtractorMOG2()  # 背景减除器
kernel = np.ones((5, 5), np.uint8)
min_area = 500

while True:
    ret, frame = cap.read()  # 读取视频帧
    if not ret:  # 如果没有读取到帧，退出循环
        print("Failed to capture video")
        break

    fgmask = fgbg.apply(frame)  # 应用背景减除器
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)  # 开操作去噪

    # 查找轮廓
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            current_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Count: {current_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv2.imshow("Frame", frame)

    # 按 'q' 键退出
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
