import torch
import cv2

# 加载模型（'yolov8n'为轻量级版本，可选yolov8s/m/l/x）
# 如果无法下载，可到相应网站下载之后再加载。
# model = torch.hub.load("ultralytics/yolov8", "yolov8n", pretrained=True)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

cap = cv2.VideoCapture(0)  # 0 是默认的摄像头
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    results = model(frame, persist=True)  # persist=True 保持追踪状态
    annotated_frame = results[0].plot()  # 绘制检测结果
    cv2.imshow("YOLO Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
