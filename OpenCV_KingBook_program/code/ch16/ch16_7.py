# ch16_7.py
import cv2
import numpy as np

src = cv2.imread("cloud.jpg")
cv2.imshow("src",src)
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)     # 影像轉成灰階
# 二值化處理影像
ret, dst_binary = cv2.threshold(src_gray,127,255,cv2.THRESH_BINARY)
# 找尋影像內的輪廓
contours, hierarchy = cv2.findContours(dst_binary,
                      cv2.RETR_LIST,
                      cv2.CHAIN_APPROX_SIMPLE)
# 取得圓中心座標和圓半徑
ellipse = cv2.fitEllipse(contours[0])               # 取得最優擬合橢圓數據
print(f"資料類型   = {type(ellipse)}")
print(f"橢圓中心   = {ellipse[0]}")
print(f"長短軸直徑 = {ellipse[1]}")
print(f"旋轉角度   = {ellipse[2]}")
dst = cv2.ellipse(src,ellipse,(0,255,0),2)          # 繪橢圓
cv2.imshow("dst",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()






