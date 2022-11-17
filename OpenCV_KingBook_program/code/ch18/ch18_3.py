# ch18_3.py
import cv2
import numpy as np

src = cv2.imread('roadtest.jpg', cv2.IMREAD_COLOR)
cv2.imshow("src", src)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)    # 轉成灰階
edges = cv2.Canny(src_gray, 50, 200)                # 使用Canny邊緣檢測
cv2.imshow("Canny", edges)                          # 顯示Canny邊緣線條
lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=10,maxLineGap=100)   
# 繪製檢測到的直線
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(src, (x1, y1), (x2, y2), (255, 0, 0), 3)   # 繪製藍色線條
cv2.imshow("dst", src)

cv2.waitKey(0)
cv2.destroyAllWindows()






