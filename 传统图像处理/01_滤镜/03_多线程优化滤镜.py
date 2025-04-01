"""
description:多线程滤镜
by:DSC
date:2025/03/31
"""

import cv2
import numpy as np
import threading
import queue
import time

# 线程安全队列（缓存3帧避免内存堆积）
frame_queue = queue.Queue(maxsize=3)
result_queue = queue.Queue(maxsize=3)


# 定义滤镜函数（示例：边缘检测）
def apply_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


# 工作线程函数
def worker_thread():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:  # 退出信号
                break
            # 处理滤镜
            processed_frame = apply_filter(frame)
            result_queue.put(processed_frame)


# 启动工作线程
worker = threading.Thread(target=worker_thread)
worker.start()

# 主线程：摄像头捕获与显示
cap = cv2.VideoCapture(0)
current_filter = apply_filter  # 当前滤镜

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 将原始帧放入队列（非阻塞式）
        if frame_queue.empty():
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # 跳过帧避免延迟累积

        # 获取处理后的帧（非阻塞）
        if not result_queue.empty():
            try:
                filtered_frame = result_queue.get_nowait()
                # 显示结果
                cv2.imshow("Filtered Video", filtered_frame)
            except queue.Empty:
                pass

        # 按Q退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # 清理资源
    frame_queue.put(None)  # 发送终止信号
    worker.join()
    cap.release()
    cv2.destroyAllWindows()
