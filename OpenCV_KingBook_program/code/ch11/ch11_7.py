# ch11_7.py
import cv2

src = cv2.imread("hung.jpg")
dst1 = cv2.blur(src,(15,15))                    # 均值濾波器
dst2 = cv2.GaussianBlur(src,(15,15),0,0)        # 高斯濾波器
dst2 = cv2.bilateralFilter(src,15,100,100)      # 雙邊濾波器

cv2.imshow("src",src)
cv2.imshow("blur",dst1)
cv2.imshow("GaussianBlur",dst1)
cv2.imshow("bilateralFilter",dst2)

cv2.waitKey(0)
cv2.destroyAllWindows()


