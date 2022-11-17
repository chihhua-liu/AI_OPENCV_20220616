# ch11_2.py
import cv2

src = cv2.imread("hung.jpg")
dst1 = cv2.blur(src, (29, 29))        # 使用 29x29 濾波核
cv2.imshow("src",src)
cv2.imshow("dst 29 x 29",dst1)

cv2.waitKey(0)
cv2.destroyAllWindows()


