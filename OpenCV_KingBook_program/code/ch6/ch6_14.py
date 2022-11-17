# ch6_14.py
import cv2
import numpy as np

img = cv2.imread("jk.jpg")              # 彩色讀取
cv2.imshow("Hung Image", img)           # 顯示影像
# ROI大小區塊建立馬賽克
face = np.random.randint(0,256,size=(190,170,3))  # 馬賽克效果
img[30:220,80:250] = face               # ROI
cv2.imshow("Face", img)                 # 顯示影像

cv2.waitKey(0)                                      
cv2.destroyAllWindows()                 # 刪除所有視窗    




 
