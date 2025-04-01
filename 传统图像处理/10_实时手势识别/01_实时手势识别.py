'''
1. ​安装依赖库
2. ​初始化 MediaPipe 手部检测模型
3. ​实时视频捕获
4. ​检测手部关键点
5. ​根据关键点判断手势
6. ​显示结果
'''

import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,        # 检测的最大手部数量
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils  # 绘图工具

def is_finger_up(hand_landmarks, finger_idx):
    """判断单个手指是否伸直"""
    tip = hand_landmarks.landmark[finger_idx[3]]  # 指尖关键点
    dip = hand_landmarks.landmark[finger_idx[2]]  # 第二关节
    mcp = hand_landmarks.landmark[finger_idx[0]]  # 掌根关键点

    # 根据关键点y坐标判断手指是否伸直（适配竖直手势）
    return tip.y < dip.y and tip.y < mcp.y

# 手指关键点索引（MediaPipe定义）
FINGERS = {
    'THUMB': [1, 2, 3, 4],
    'INDEX': [5, 6, 7, 8],
    'MIDDLE': [9, 10, 11, 12],
    'RING': [13, 14, 15, 16],
    'PINKY': [17, 18, 19, 20]
}

cap = cv2.VideoCapture(0)  # 打开摄像头

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # 转换为RGB格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制手部关键点
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 判断各手指状态
            fingers_up = {
                'thumb': is_finger_up(hand_landmarks, FINGERS['THUMB']),
                'index': is_finger_up(hand_landmarks, FINGERS['INDEX']),
                'middle': is_finger_up(hand_landmarks, FINGERS['MIDDLE']),
                'ring': is_finger_up(hand_landmarks, FINGERS['RING']),
                'pinky': is_finger_up(hand_landmarks, FINGERS['PINKY'])
            }

            # 统计伸直的手指数量
            count = sum(fingers_up.values())

            # 根据数量显示手势
            cv2.putText(frame, f'Count: {count}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()