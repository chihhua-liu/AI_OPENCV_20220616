# ch12_20.py
import cv2
import numpy as np

src = cv2.imread("excel.jpg")
kernel = np.ones((11,11),np.uint8)                      # 建立11x11內核
dst = cv2.morphologyEx(src,cv2.MORPH_BLACKHAT,kernel)   # blackhat

cv2.imshow("src",src)
cv2.imshow("after blackhat",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()



