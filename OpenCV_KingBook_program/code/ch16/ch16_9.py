# ch16_9.py
import cv2
import numpy as np

src = cv2.imread("multiple.jpg")
cv2.imshow("src",src)
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)     # 影像轉成灰階
# 二值化處理影像
ret, dst_binary = cv2.threshold(src_gray,127,255,cv2.THRESH_BINARY)
# 找尋影像內的輪廓
contours, hierarchy = cv2.findContours(dst_binary,
                      cv2.RETR_LIST,
                      cv2.CHAIN_APPROX_SIMPLE)
# 近似多邊形包圍
n = len(contours)                                   # 輪廓數量
src1 = src.copy()                                   # 複製src影像
src2 = src.copy()                                   # 複製src影像
for i in range(n):
    approx = cv2.approxPolyDP(contours[i], 3, True)         # epsilon=3
    dst1 = cv2.polylines(src1,[approx],True,(0,255,0),2)    # dst1
    approx = cv2.approxPolyDP(contours[i], 15, True)        # epsilon=15
    dst2 = cv2.polylines(src2,[approx],True,(0,255,0),2)    # dst2
cv2.imshow("dst1 - epsilon = 3",dst1)
cv2.imshow("dst2 - epsilon = 15",dst2)

cv2.waitKey(0)
cv2.destroyAllWindows()






