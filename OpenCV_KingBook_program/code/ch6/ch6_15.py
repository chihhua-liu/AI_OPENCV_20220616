# ch6_15.py
import cv2
import numpy as np

img = cv2.imread("jk.jpg")              # 彩色讀取
cv2.imshow("Hung Image", img)           # 顯示影像
usa = cv2.imread("money.jpg")           # 彩色讀取
cv2.imshow("Money Image", usa)          # 顯示影像
face = img[30:220,80:250]               # ROI
usa[30:220,120:290] = face              # 複製到usa影像
cv2.imshow("Image", usa)                # 顯示影像

cv2.waitKey(0)                                      
cv2.destroyAllWindows()                 # 刪除所有視窗    




 
