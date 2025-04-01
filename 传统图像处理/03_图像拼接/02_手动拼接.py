"""
# 这段代码不能正常运行，需要修改。
import cv2
import numpy as np

def manual_stitching(img1, img2):
    # 特征检测与匹配
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches)*0.15)]

    # 计算Homography矩阵
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 计算画布大小
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners2 = np.array([[0,0], [0,h2], [w2,h2], [w2,0]], dtype=np.float32).reshape(-1,1,2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)
    all_corners = np.concatenate((np.array([[0,0], [w1,0], [0,h1], [w1,h1]]), warped_corners2.reshape(4,2)))
    x_min, y_min = np.floor(np.min(all_corners, axis=0)).astype(int)
    x_max, y_max = np.ceil(np.max(all_corners, axis=0)).astype(int)
    canvas_size = (x_max - x_min, y_max - y_min)

    # 图像变换
    transform = np.array([[1,0,-x_min], [0,1,-y_min], [0,0,1]])
    warped_img1 = cv2.warpPerspective(img1, transform, canvas_size)
    warped_img2 = cv2.warpPerspective(img2, transform.dot(H), canvas_size)

    # 加权融合
    mask1 = (warped_img1 > 0).astype(np.uint8)
    mask2 = (warped_img2 > 0).astype(np.uint8)
    overlap = cv2.bitwise_and(mask1, mask2)
    result = warped_img1.copy()
    result[overlap == 1] = 0.5 * warped_img1[overlap == 1] + 0.5 * warped_img2[overlap == 1]
    result = np.where(warped_img2 > 0, warped_img2, result)

    return result

# 执行拼接
img1 = cv2.imread('image2.jpg')
img2 = cv2.imread('image1.jpg')
panorama = manual_stitching(img1, img2)
cv2.imwrite('panorama_manual.jpg', panorama)

"""
