"""
注意这种方法只能拼接两张图像，且两张图像必须有重叠部分。重叠部分最好大于30%。
图像输入顺序必须是从左到右，不能是从右到左。
如果需要拼接多张图像，可以使用循环来实现。
"""

import cv2

# 读取图像
img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

# 创建Stitcher对象
stitcher = cv2.Stitcher_create()

# 执行拼接
status, result = stitcher.stitch([img1, img2])

if status == cv2.Stitcher_OK:
    cv2.imwrite("panorama.jpg", result)
else:
    print("拼接失败！错误代码:", status)
