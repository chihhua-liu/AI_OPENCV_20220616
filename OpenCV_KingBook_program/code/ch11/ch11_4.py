# ch11_4.py
import cv2

src = cv2.imread("hung.jpg")
dst1 = cv2.medianBlur(src, 3)        # 使用邊長是 3 的濾波核
dst2 = cv2.medianBlur(src, 5)        # 使用邊長是 5 的濾波核
dst3 = cv2.medianBlur(src, 7)        # 使用邊長是 7 的濾波核
cv2.imshow("src",src)
cv2.imshow("dst 3 x 3",dst1)
cv2.imshow("dst 5 x 5",dst2)
cv2.imshow("dst 7 x 7",dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()


