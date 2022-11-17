# ch11_2_1.py
import cv2

src = cv2.imread("hung.jpg")
dst1 = cv2.boxFilter(src,-1,(2,2),normalize=0)  # ksize是 2x2 的濾波核
dst2 = cv2.boxFilter(src,-1,(3,3),normalize=0)  # ksize是 3x3 的濾波核
dst3 = cv2.boxFilter(src,-1,(5,5),normalize=0)  # ksize是 5x5 的濾波核
cv2.imshow("src",src)
cv2.imshow("dst 2 x 2",dst1)
cv2.imshow("dst 3 x 3",dst2)
cv2.imshow("dst 5 x 5",dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()


