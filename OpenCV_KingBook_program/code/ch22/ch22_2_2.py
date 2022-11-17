# ch22_2_2.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft JhengHei"]

src = cv2.imread('coin1.jpg',cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# 因為在matplotlib模組顯示, 所以必須轉成 RGB 色彩
rgb_src = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)       

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8)
# 執行開運算 Opening
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
# 獲得距離轉換函數結果
dst = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# 讀者也可以更改下列 0.7 為其他值, 會影響前景大小
ret, sure_fg = cv2.threshold(dst,0.7*dst.max(),255,0)  # 前景圖案
plt.subplot(131)
plt.title("原始影像")
plt.imshow(rgb_src)
plt.axis('off')
plt.subplot(132)
plt.title("距離變換影像")
plt.imshow(dst)
plt.axis('off')
plt.subplot(133)
plt.title("閾值化影像")
plt.imshow(sure_fg)
plt.axis('off')
plt.show()






