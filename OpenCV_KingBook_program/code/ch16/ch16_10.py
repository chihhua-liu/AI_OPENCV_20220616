# ch16_10.py
import cv2
import numpy as np

src = cv2.imread("unregular.jpg")
cv2.imshow("src",src)
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)     # 影像轉成灰階
# 二值化處理影像
ret, dst_binary = cv2.threshold(src_gray,127,255,cv2.THRESH_BINARY)
# 找尋影像內的輪廓
contours, hierarchy = cv2.findContours(dst_binary,
                      cv2.RETR_LIST,
                      cv2.CHAIN_APPROX_SIMPLE)
# 擬合一條線
rows, cols = src.shape[:2]                          # 輪廓大小
vx,vy,x,y = cv2.fitLine(contours[0],cv2.DIST_L2,0,0.01,0.01)
print(f"共線正規化向量 = {vx}, {vy}")
print(f"直線經過的點   = {x}, {y}")
lefty = int((-x * vy / vx) + y)                     # 左邊點的 y 座標
righty = int(((cols - x) * vy / vx) + y)            # 右邊點的 y 座標
dst = cv2.line(src,(0,lefty),(cols-1,righty),(0,255,0),2)   # 左到右繪線             
cv2.imshow("dst",dst)           

cv2.waitKey(0)
cv2.destroyAllWindows()






