"""
运行之前需要先正确安装以下库：
- OpenCV
- NumPy
- pytesseract
- tesseract-ocr
- tesseract-ocr-chi-sim (简体中文支持)

要完整运行并正确识别车牌，可能还需要修改一些参数和配置。
"""

import cv2
import numpy as np
import pytesseract

# ---------------------------- 配置参数 ----------------------------
DEBUG = True  # 调试模式开关
RESIZE_WIDTH = 800  # 处理图像宽度
MIN_ASPECT_RATIO = 1.8  # 车牌最小宽高比
MAX_ASPECT_RATIO = 8.0  # 车牌最大宽高比
COLOR_THRESHOLD = 0.15  # 颜色验证阈值

# HSV颜色空间范围（蓝/黄/绿）
COLOR_RANGES = {
    "blue": ([90, 80, 80], [140, 255, 255]),
    "yellow": ([15, 80, 80], [40, 255, 255]),
    "green": ([40, 40, 40], [90, 255, 255]),
}


# ---------------------------- 工具函数 ----------------------------
def debug_show(name, img):
    """调试显示函数"""
    if DEBUG:
        h, w = img.shape[:2]
        show_img = cv2.resize(img, (w // 2, h // 2)) if w > 600 else img
        cv2.imshow(name, show_img)
        cv2.waitKey(0)


def adaptive_resize(img):
    """保持宽高比的智能缩放"""
    h, w = img.shape[:2]
    scale = RESIZE_WIDTH / w
    return cv2.resize(img, (RESIZE_WIDTH, int(h * scale)))


def enhance_details(img):
    """细节增强"""
    return cv2.detailEnhance(img, sigma_s=15, sigma_r=0.2)


# ---------------------------- 核心算法 ----------------------------
def locate_license_plate(img):
    """鲁棒性车牌定位算法"""
    # 预处理
    img = adaptive_resize(img)
    enhanced = enhance_details(img)
    debug_show("1. Enhanced", enhanced)

    # 多颜色空间检测
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    color_masks = {
        color: cv2.inRange(hsv, np.array(low), np.array(up))
        for color, (low, up) in COLOR_RANGES.items()
    }

    # 选择主颜色通道
    main_color = max(color_masks, key=lambda x: cv2.countNonZero(color_masks[x]))
    mask = color_masks[main_color]
    debug_show(f"2. {main_color} Mask", mask)

    # 改进的形态学处理（保留字符间隙）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 3))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    debug_show("3. Morph Result", closed)

    # 轮廓检测
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 候选区域筛选
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        area = w * h

        # 基础几何条件
        if not (
            MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO and 1500 < area < 80000
        ):
            continue

        # 颜色二次验证
        roi = enhanced[y : y + h, x : x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_score = cv2.countNonZero(
            cv2.inRange(
                hsv_roi,
                np.array(COLOR_RANGES[main_color][0]),
                np.array(COLOR_RANGES[main_color][1]),
            )
        ) / (w * h)

        if color_score > COLOR_THRESHOLD:
            # 动态评分公式
            score = (area / 1000) * (1 - abs(aspect_ratio - 3.5)) * color_score
            candidates.append((score, x, y, w, h))

    # 调试显示候选区域
    if DEBUG:
        debug_img = enhanced.copy()
        for _, x, y, w, h in candidates:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        debug_show("4. Candidates", debug_img)

    # 选择最佳候选
    if candidates:
        # 按评分排序取前三
        candidates.sort(reverse=True, key=lambda x: x[0])
        top3 = candidates[:3]

        # 选择最接近图像中心的候选
        img_center = np.array([enhanced.shape[1] // 2, enhanced.shape[0] // 2])
        distances = []
        for score, x, y, w, h in top3:
            box_center = np.array([x + w // 2, y + h // 2])
            distances.append(np.linalg.norm(box_center - img_center))

        best_idx = np.argmin(distances)
        best = top3[best_idx]
        return enhanced[best[2] : best[2] + best[4], best[1] : best[1] + best[3]]

    return None


def recognize_characters(plate):
    """字符识别"""
    # 预处理
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 垂直投影分割
    projection = np.sum(binary, axis=0)
    threshold = np.max(projection) * 0.1
    segments = np.where(projection > threshold)[0]

    # 分割字符
    char_imgs = []
    if len(segments) > 0:
        start = segments[0]
        for i in range(1, len(segments)):
            if segments[i] - segments[i - 1] > 2:
                char_imgs.append(binary[:, start : segments[i - 1]])
                start = segments[i]
        char_imgs.append(binary[:, start : segments[-1]])

    # OCR识别
    config = (
        "--psm 8 --oem 3 "
        "-c tessedit_char_whitelist=京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新使ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    result = []
    for char in char_imgs:
        char = cv2.resize(char, (30, 60))
        char = cv2.medianBlur(char, 3)
        text = pytesseract.image_to_string(char, config=config)
        result.append(text.strip())

    return "".join(result).replace(" ", "").upper()


# ---------------------------- 主流程 ----------------------------
def main(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("错误：无法读取图像文件")
        return

    # 车牌定位
    plate = locate_license_plate(img)
    if plate is None:
        print("未检测到车牌区域")
        return

    # 字符识别
    plate_number = recognize_characters(plate)
    print(f"识别结果: {plate_number}")

    # 显示结果
    cv2.imshow("Detected Plate", plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("car.jpg")
