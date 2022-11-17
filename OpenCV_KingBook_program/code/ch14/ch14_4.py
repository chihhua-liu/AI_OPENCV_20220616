# ch14_4.py
import cv2

src = cv2.imread("pengiun.jpg")         # 讀取影像
dst1 = src + src                        # 影像相加
dst2 = src - src                        # 影像相減
cv2.imshow("src",src)
cv2.imshow("dst1 - add",dst1)
cv2.imshow("dst2 - subtraction",dst2)

cv2.waitKey(0)
cv2.destroyAllWindows()
