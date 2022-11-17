# ch12_15.py
import cv2
import numpy as np

src = cv2.imread("k.jpg")
kernel = np.ones((5,5),np.uint8)                # 建立5x5內核
dst1 = cv2.dilate(src, kernel)                  # dilation
dst2 = cv2.erode(src, kernel)                   # erosion
cv2.imshow("src",src)
cv2.imshow("after dilation 5 x 5",dst1)
cv2.imshow("after erosion 5 x 5",dst2)

cv2.waitKey(0)
cv2.destroyAllWindows()



