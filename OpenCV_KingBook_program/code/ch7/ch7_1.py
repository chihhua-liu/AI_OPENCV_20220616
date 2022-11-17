# ch7_1.py
import cv2
import numpy as np

img = np.ones((350,500,3),np.uint8) * 255       # 建立白色底的畫布
cv2.line(img,(1,1),(300,1),(255,0,0))           # 上方水平直線
cv2.line(img,(300,1),(300,300),(255,0,0))       # 右邊垂直直線
cv2.line(img,(300,300),(1,300),(255,0,0))       # 下邊水平直線
cv2.line(img,(1,300),(1,1),(255,0,0))           # 左邊垂直直線
cv2.imshow("My Draw",img)                       # 畫布顯示直線

cv2.waitKey(0)                                      
cv2.destroyAllWindows()                         # 刪除所有視窗






 
