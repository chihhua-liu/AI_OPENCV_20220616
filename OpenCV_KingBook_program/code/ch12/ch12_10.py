# ch12_10.py
import cv2
import numpy as np

src = cv2.imread("night.jpg")
kernel = np.ones((9,9),np.uint8)                    # 建立9x9內核
dst = cv2.morphologyEx(src,cv2.MORPH_OPEN,kernel)   # 開運算

cv2.imshow("src",src)
cv2.imshow("after Opening 9 x 9",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()





