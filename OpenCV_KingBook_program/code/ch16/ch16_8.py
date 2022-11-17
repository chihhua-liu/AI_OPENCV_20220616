# ch16_8.py
import cv2
import numpy as np

src = cv2.imread("heart.jpg")
cv2.imshow("src",src)
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)     # 影像轉成灰階
# 二值化處理影像
ret, dst_binary = cv2.threshold(src_gray,127,255,cv2.THRESH_BINARY)
# 找尋影像內的輪廓
contours, hierarchy = cv2.findContours(dst_binary,
                      cv2.RETR_LIST,
                      cv2.CHAIN_APPROX_SIMPLE)
# 取得三角形面積與頂點座標
area, triangle = cv2.minEnclosingTriangle(contours[0])  
print(f"三角形面積   = {area}")
print(f"三角形頂點座標資料類型 = {type(triangle)}")
print(f"三角頂點座標 = \n{triangle}")
triangle = np.int0(triangle)                        # 轉整數
dst = cv2.line(src,tuple(triangle[0][0]),tuple(triangle[1][0]),(0,255,0),2)
dst = cv2.line(src,tuple(triangle[1][0]),tuple(triangle[2][0]),(0,255,0),2)
dst = cv2.line(src,tuple(triangle[0][0]),tuple(triangle[2][0]),(0,255,0),2)               
cv2.imshow("dst",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()






