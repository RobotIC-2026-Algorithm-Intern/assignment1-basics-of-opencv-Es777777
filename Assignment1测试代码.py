# 传统视觉常用起手式
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 读取图像，OpenCV的imread函数有第二个参数，表示以何种方式读取图像
# cv2.IMREAD_COLOR：以彩色图像方式读取，忽略alpha通道，默认方式
# cv2.IMREAD_GRAYSCALE：以灰度图像方式读取
# cv2.IMREAD_UNCHANGED：包含alpha通道，以包含alpha通道方式读取
# 这里的图像路径不能包含中文，否则会报错

tsubaki = cv2.imread('res/tsubaki.png')
kita = cv2.imread('res/kita.png')
lena = cv2.imread('res/lena.png', cv2.IMREAD_GRAYSCALE)
# OpenCV读取图像时，通道顺序是BGR，而不是RGB

# 可以使用Python中的type函数查看对象的类型，证明Python中的OpenCV，使用numpy的ndarray存储图像
print(type(lena), type(kita))

# numpy的ndarray本质上是张量（Tensor），提供一个shape属性，告诉我们图像的尺寸（row, column, channel）。
print(lena.shape, kita.shape)

# numpy的ndarray提供一个dtype属性，告诉我们像素是以何种数据类型被存储的
print(lena.dtype, kita.dtype)
cv2.imshow('kita', kita)
cv2.waitKey(0)

kita_grey = cv2.cvtColor(kita, cv2.COLOR_BGR2GRAY)
# 同时也有cv2.COLOR_GRAY2BGR参数进行逆转换，其转换方法为R = G = B = Y
cv2.imshow('kita_grey', kita_grey)
cv2.waitKey(0)

cv2.imwrite('kita_grey.png', kita_grey)
# 设置图像大小
plt.rcParams['figure.figsize'] = (12, 8)

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
            plt.imshow(img[:, :, ::-1]) # 对[:, :, ::-1]有疑惑的同学可以研究一下Python的切片