# ch7_25.py
import cv2
import numpy as np

def onChange(x):
    b = cv2.getTrackbarPos("B",'canvas')            # 建立B通道顏色
    g = cv2.getTrackbarPos("G",'canvas')            # 建立G通道顏色
    r = cv2.getTrackbarPos("R",'canvas')            # 建立R通道顏色
    canvas[:] = [b,g,r]                             # 設定背景色

canvas = np.ones((200,640,3),np.uint8) * 255        # 寬640,高200
cv2.namedWindow("canvas")
cv2.createTrackbar("B","canvas",0,255,onChange)     # 藍色通道控制
cv2.createTrackbar("G","canvas",0,255,onChange)     # 綠色通道控制
cv2.createTrackbar("R","canvas",0,255,onChange)     # 紅色通道控制                  
while 1:
    cv2.imshow("canvas",canvas)
    key = cv2.waitKey(100)                      # 0.1秒檢查一次
    if key == 27:                               # Esc 則結束
        break
   
cv2.destroyAllWindows()                         # 刪除所有視窗













 
