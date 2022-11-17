# ch16_4.py
import cv2
import numpy as np

src = cv2.imread("explode2.jpg")
cv2.imshow("src",src)
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)         # 影像轉成灰階
# 二值化處理影像
ret, dst_binary = cv2.threshold(src_gray,127,255,cv2.THRESH_BINARY)
# 找尋影像內的輪廓
contours, hierarchy = cv2.findContours(dst_binary,
                      cv2.RETR_LIST,
                      cv2.CHAIN_APPROX_SIMPLE)  

box = cv2.minAreaRect(contours[0])                      # 建構最小矩形
print(f"轉換前的矩形頂角 = \n {box}")
points = cv2.boxPoints(box)                             # 獲取頂點座標
points = np.int0(points)                                # 轉為整數
print(f"轉換後的矩形頂角 = \n {points}")
dst = cv2.drawContours(src,[points],0,(0,255,0),2)      # 繪製輪廓
cv2.imshow("dst",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()






