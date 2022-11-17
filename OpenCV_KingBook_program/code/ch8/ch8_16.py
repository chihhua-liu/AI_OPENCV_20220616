# ch8_16.py
import cv2
import numpy as np

src = cv2.imread("forest.jpg")          # 讀取影像
dst = cv2.bitwise_not(src)              # 執行or運算
cv2.imshow("Forest",src)
cv2.imshow("Not Forest",dst)

cv2.waitKey()
cv2.destroyAllWindows()                 # 刪除所有視窗






