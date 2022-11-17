# ch12_11.py
import cv2
import numpy as np

src = cv2.imread("night.jpg")
kernel = np.ones((9,9),np.uint8)                    # 建立9x9內核
mid = cv2.erode(src, kernel)                        # erosion
dst = cv2.dilate(mid, kernel)                       # dilation

cv2.imshow("src",src)
cv2.imshow("after erosion 9 x 9",mid)
cv2.imshow("after dilation 9 x 9",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()



