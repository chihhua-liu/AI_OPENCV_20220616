# ch11_5.py
import cv2

src = cv2.imread("hung.jpg")
dst1 = cv2.GaussianBlur(src,(3,3),0,0)      # 使用 3 x 3 的濾波核
dst2 = cv2.GaussianBlur(src,(5,5),0,0)      # 使用 5 x 5 的濾波核
dst3 = cv2.GaussianBlur(src,(29,29),0,0)    # 使用 29 x 29 的濾波核
cv2.imshow("src",src)
cv2.imshow("dst 3 x 3",dst1)
cv2.imshow("dst 5 x 5",dst2)
cv2.imshow("dst 15 x 15",dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()


