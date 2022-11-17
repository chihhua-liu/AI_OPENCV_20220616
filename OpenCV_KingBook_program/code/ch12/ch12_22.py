# ch12_22.py
import cv2
import numpy as np

src = cv2.imread("bw_circle.jpg")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(39,39))
dst1 = cv2.dilate(src, kernel)          
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(39,39))
dst2 = cv2.dilate(src, kernel)          
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(39,39))
dst3 = cv2.dilate(src, kernel)          

cv2.imshow("src",src)
cv2.imshow("MORPH_RECT",dst1)
cv2.imshow("MORPH_ELLIPSE",dst2)
cv2.imshow("MORPH_CROSS",dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()





