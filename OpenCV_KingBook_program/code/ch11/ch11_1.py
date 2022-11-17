# ch11_1.py
import cv2

src = cv2.imread("hung.jpg")
dst1 = cv2.blur(src, (3, 3))        # 使用 3x3 濾波核
dst2 = cv2.blur(src, (5, 5))        # 使用 5x5 濾波核
dst3 = cv2.blur(src, (7, 7))        # 使用 7x7 濾波核
cv2.imshow("src",src)
cv2.imshow("dst 3 x 3",dst1)
cv2.imshow("dst 5 x 5",dst2)
cv2.imshow("dst 7 x 7",dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()


