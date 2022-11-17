# ch11_6.py
import cv2

src = cv2.imread("border.jpg")
dst1 = cv2.blur(src, (3, 3))            # 均值濾波器 - 3x3 濾波核
dst2 = cv2.blur(src, (7, 7))            # 均值濾波器 - 7x7 濾波核

dst3 = cv2.GaussianBlur(src,(3,3),0,0)  # 高斯濾波器 - 3x3 的濾波核
dst4 = cv2.GaussianBlur(src,(7,7),0,0)  # 高斯濾波器 - 7x7 的濾波核

cv2.imshow("dst 3 x 3",dst1)
cv2.imshow("dst 7 x 7",dst2)
cv2.imshow("Gauss dst 3 x 3",dst3)
cv2.imshow("Gauss dst 7 x 7",dst4)

cv2.waitKey(0)
cv2.destroyAllWindows()


