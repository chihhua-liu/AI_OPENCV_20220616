# ch18_4.py
import cv2
import numpy as np

src = cv2.imread('shapes.jpg')
cv2.imshow("src", src)
image = cv2.medianBlur(src,5)                           # 過濾雜訊
src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      # 轉成灰階
circles = cv2.HoughCircles(src_gray,cv2.HOUGH_GRADIENT,1,100,param1=50,
                           param2=30,minRadius=70,maxRadius=200)
circles = np.uint(np.around(circles))                   # 轉成整數
# 繪製檢測到的直線
for c in circles[0]:
    x, y, r = c
    cv2.circle(src,(x, y), r, (0,255,0),3)              # 綠色繪圓外圈
    cv2.circle(src,(x, y), 2, (0,0,255),2)              # 紅色繪圓中心
cv2.imshow("dst", src)

cv2.waitKey(0)
cv2.destroyAllWindows()






