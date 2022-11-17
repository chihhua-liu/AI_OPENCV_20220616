# ch22_1.py
import cv2
import numpy as np

src = cv2.imread('coin1.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow("Src", src)
ret, dst = cv2.threshold(src,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("Dst", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()



