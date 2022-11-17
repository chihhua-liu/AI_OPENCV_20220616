# ch12_14.py
import cv2
import numpy as np

src = cv2.imread("night.jpg")
kernel = np.ones((9,9),np.uint8)                # 建立9x9內核
mid = cv2.dilate(src, kernel)                   # dilation
dst = cv2.erode(mid, kernel)                    # erosion
cv2.imshow("src",src)
cv2.imshow("after dilation 9 x 9",mid)
cv2.imshow("after erosion 9 x 9",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()



