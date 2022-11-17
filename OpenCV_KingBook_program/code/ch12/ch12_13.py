# ch12_13.py
import cv2
import numpy as np

src = cv2.imread("snowman1.jpg")
kernel = np.ones((11,11),np.uint8)                  # 建立11x11內核
dst = cv2.morphologyEx(src,cv2.MORPH_CLOSE,kernel)  # 閉運算

cv2.imshow("src",src)
cv2.imshow("after Closing 11 x 11",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()





