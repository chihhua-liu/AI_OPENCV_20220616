# ch7_10.py
import cv2
import numpy as np

height = 400                            # 畫布高度
width = 600                             # 畫布寬度
img = np.zeros((height,width,3),np.uint8)   # 建立黑底畫布陣列
for i in range(0,50):
    cx = np.random.randint(0,width)     # 隨機數圓心的 x 軸座標
    cy = np.random.randint(0,height)    # 隨機數圓心的 y 軸座標
    color = np.random.randint(0,256, size=3).tolist() # 建立隨機色彩
    r = np.random.randint(5,100)        # 在5 - 100間的隨機半徑
    cv2.circle(img,(cx,cy),r,color,-1)  # 建立隨機實心圓
cv2.imshow("Random Circle",img)

cv2.waitKey(0)                                      
cv2.destroyAllWindows()                 # 刪除所有視窗














 
