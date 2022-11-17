# ch12_18.py
import cv2
import numpy as np

src = cv2.imread("btree.jpg")
kernel = np.ones((3,3),np.uint8)                    # 建立3x3內核
dst = cv2.morphologyEx(src,cv2.MORPH_TOPHAT,kernel) # tophat

cv2.imshow("src",src)
cv2.imshow("after tophat",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()



