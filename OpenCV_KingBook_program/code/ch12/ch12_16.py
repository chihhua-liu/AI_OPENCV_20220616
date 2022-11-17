# ch12_16.py
import cv2
import numpy as np

src = cv2.imread("k.jpg")
kernel = np.ones((5,5),np.uint8)                        # 建立5x5內核
dst = cv2.morphologyEx(src,cv2.MORPH_GRADIENT,kernel)   # gradient

cv2.imshow("src",src)
cv2.imshow("after morpological gradient",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()



