# ch8_10.py
import cv2
import numpy as np

src1 = cv2.imread("lake.jpg")                       # 影像 src1
cv2.imshow("lake",src1)
src2 = cv2.imread("geneva.jpg")                     # 影像 src2
cv2.imshow("geneva.jpg",src2)
alpha = 1
beta = 0.2
gamma = 1
dst = cv2.addWeighted(src1,alpha,src2,beta,gamma)   # 加權和
cv2.imshow("lake+geneva",dst)                       # 顯示結果

cv2.waitKey()
cv2.destroyAllWindows()



