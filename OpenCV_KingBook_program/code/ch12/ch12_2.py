# ch12_2.py
import cv2
import numpy as np

src = cv2.imread("bw.jpg")
kernel = np.ones((5, 5),np.uint8)       # 建立5x5內核
dst1 = cv2.erode(src, kernel)           # 腐蝕操作
kerne2 = np.ones((11, 11),np.uint8)     # 建立11x11內核
dst2 = cv2.erode(src, kerne2)           # 腐蝕操作

cv2.imshow("src",src)
cv2.imshow("after erosion 5 x 5",dst1)
cv2.imshow("after erosion 11 x 11",dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()





