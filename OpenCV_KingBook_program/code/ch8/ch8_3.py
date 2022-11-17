# ch8_3.py
import cv2
import numpy as np

img = cv2.imread("jk.jpg")                  # 彩色讀取
res = cv2.add(img, img)                     # 調整亮度結果
cv2.imshow("MyPicture1", img)               # 顯示影像img
cv2.imshow("MyPicture2", res)               # 顯示影像res

cv2.waitKey(0)                              # 等待
cv2.destroyAllWindows()                     # 刪除所有視窗












 
