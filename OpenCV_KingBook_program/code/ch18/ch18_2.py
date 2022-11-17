# ch18_2.py
import cv2
import numpy as np

src = cv2.imread('lane.jpg', cv2.IMREAD_COLOR)
cv2.imshow("src", src)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)    # 轉成灰階
edges = cv2.Canny(src_gray, 100, 200)               # 使用Canny邊緣檢測
#cv2.imshow("Canny", edges)                         # 顯示Canny邊緣線條
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)    # 檢測直線
# 繪製直線
for line in lines:
    rho, theta = line[0]                            # lines回傳
    a = np.cos(theta)                               # cos(theta)
    b = np.sin(theta)                               # sin(theta)
    x0 = rho * a
    y0 = rho * b
    x1 = int(x0 + 1000*(-b))                        # 建立 x1
    y1 = int(y0 + 1000*(a))                         # 建立 y1
    x2 = int(x0 - 1000*(-b))                        # 建立 x2
    y2 = int(y0 - 1000*(a))                         # 建立 y2
    cv2.line(src,(x1,y1),(x2,y2),(0,0,255),2)       # 繪製紅色線條
cv2.imshow("dst", src)

cv2.waitKey(0)
cv2.destroyAllWindows()






