# ch12_3.py
import cv2
import numpy as np

src = cv2.imread("bw_noise.jpg")
kernel = np.ones((3,3),np.uint8)        # 建立3x3內核
dst1 = cv2.erode(src, kernel)           # 腐蝕操作
kerne2 = np.ones((5,5),np.uint8)        # 建立5x5內核
dst2 = cv2.erode(src, kerne2)           # 腐蝕操作

cv2.imshow("src",src)
cv2.imshow("after erosion 3 x 3",dst1)
cv2.imshow("after erosion 5 x 5",dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()





