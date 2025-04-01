import cv2

# 加载预训练的Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取输入图像
img = cv2.imread('lena.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图

# 人脸检测
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,  # 图像缩放因子
    minNeighbors=5,   # 检测结果的保留阈值
    minSize=(30, 30)  # 最小人脸尺寸
)

# 在图像上画矩形框
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 显示结果
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()