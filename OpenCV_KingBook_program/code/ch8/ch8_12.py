# ch8_12.py
import cv2
import numpy as np

src1 = cv2.imread("jk.jpg",cv2.IMREAD_GRAYSCALE)    # 讀取影像
src2 = np.zeros(src1.shape,dtype=np.uint8)          # 建立mask

src2[30:260,70:260]=255
dst = cv2.bitwise_and(src1,src2)                    # 執行and運算
cv2.imshow("Hung",src1)
cv2.imshow("Mask",src2)
cv2.imshow("Result",dst)

cv2.waitKey()
cv2.destroyAllWindows()                             # 刪除所有視窗





