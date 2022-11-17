# ch11_8.py
import cv2
import numpy as np

src = cv2.imread("hung.jpg")
kernel = np.ones((11,11),np.float32) / 121  # 自訂卷積核
dst = cv2.filter2D(src,-1,kernel)           # 自定義濾波器
cv2.imshow("src",src)
cv2.imshow("dst",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()


