# ch12_7.py
import cv2
import numpy as np

src = cv2.imread("a.jpg")
kernel = np.ones((3, 3),np.uint8)       # 建立3x3內核
dst1 = cv2.dilate(src, kernel)          # 膨脹操作
kerne2 = np.ones((5, 5),np.uint8)       # 建立5x5內核
dst2 = cv2.dilate(src, kerne2)          # 膨脹操作

cv2.imshow("src",src)
cv2.imshow("after dilation 3 x 3",dst1)
cv2.imshow("after dilation 5 x 5",dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()





