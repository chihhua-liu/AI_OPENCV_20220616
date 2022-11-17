# ch16_6.py
import cv2
import numpy as np

src = cv2.imread("explode1.jpg")
cv2.imshow("src",src)
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)     # 影像轉成灰階
# 二值化處理影像
ret, dst_binary = cv2.threshold(src_gray,127,255,cv2.THRESH_BINARY)
# 找尋影像內的輪廓
contours, hierarchy = cv2.findContours(dst_binary,
                      cv2.RETR_LIST,
                      cv2.CHAIN_APPROX_SIMPLE)  
# 取得圓中心座標和圓半徑
(x, y), radius = cv2.minEnclosingCircle(contours[0])    
center = (int(x), int(y))                           # 圓中心座標取整數
radius = int(radius)                                # 圓半徑取整數
dst = cv2.circle(src,center,radius,(0,255,255),2)   # 繪圓
cv2.imshow("dst",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()






