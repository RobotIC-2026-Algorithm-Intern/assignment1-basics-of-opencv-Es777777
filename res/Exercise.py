import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_img(*imgs: np.ndarray) -> None:
    """使用matplotlib在Notebook中绘制图像"""
    plt.figure()
    for idx, img in enumerate(imgs):
        plt.subplot(1, len(imgs), idx + 1)
        if len(img.shape) == 2:
            # 灰度图
            plt.imshow(img, cmap='gray')
        elif len(img.shape) == 3:
            # 彩色图在绘制时需要将OpenCV的BGR格式转为RGB格式
            plt.imshow(img[:, :, ::-1])# 对[:, :, ::-1]有疑惑的同学可以研究一下Python的切片

img = cv2.imread('ex1.jpg')
show_img(img)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # BGR转HSV

blue_l = np.array([100, 80, 0])
blue_u = np.array([140, 255, 255])
b_mask = cv2.inRange(img_hsv, blue_l, blue_u)  #生成蓝色掩码

kernel = np.ones((18, 18), np.uint8)  # 可以调整核大小
mask_clean = cv2.morphologyEx(b_mask, cv2.MORPH_OPEN, kernel)

img_copy = img.copy()
img_b, img_g, img_r = cv2.split(img)  #分离BGR通道
img_copy[mask_clean == 0] = 0  #将非蓝色部分像素值设为0
img_b[mask_clean == 0] = 0  #将非蓝色部分像素值设为0
show_img(img, img_b, img_copy)

show_img(img_copy)