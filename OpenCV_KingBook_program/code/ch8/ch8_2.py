# ch8_2.py
import cv2
import numpy as np

img = cv2.imread("jk.jpg", cv2.IMREAD_GRAYSCALE)    # 灰色讀取
res = cv2.add(img, img)
cv2.imshow("MyPicture1", img)                       # 顯示影像img
cv2.imshow("MyPicture2", res)                       # 顯示影像res

cv2.waitKey(0)                                      # 等待
cv2.destroyAllWindows()                             # 刪除所有視窗












 
