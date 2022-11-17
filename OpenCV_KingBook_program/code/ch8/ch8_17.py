# ch8_17.py
import cv2
import numpy as np

src1 = cv2.imread("forest.jpg")         # 讀取影像
src2 = np.zeros(src1.shape,np.uint8)

src2[:,120:360,:] = 255                 # 建立mask白色區塊
dst = cv2.bitwise_xor(src1,src2)        # 執行xor運算
cv2.imshow("Forest",src1)               # forest.jpg
cv2.imshow("Mask",src2)                 # mask
cv2.imshow("Forest xor operation",dst)  # 結果

cv2.waitKey()
cv2.destroyAllWindows()                 # 刪除所有視窗






