# ch12_9.py
import cv2
import numpy as np

src = cv2.imread("btree.jpg")
kernel = np.ones((3,3),np.uint8)                    # 建立3x3內核
dst = cv2.morphologyEx(src,cv2.MORPH_OPEN,kernel)   # 開運算

cv2.imshow("src",src)
cv2.imshow("after Opening 3 x 3",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()





