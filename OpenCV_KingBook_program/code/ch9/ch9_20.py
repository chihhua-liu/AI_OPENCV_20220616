# ch9_20.py
import cv2

src = cv2.imread("jk.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("Src",src)
thresh = 127                        # 定義閾值 = 127
maxval = 255                        # 定義像素最大值
ret, dst = cv2.threshold(src,thresh,maxval,cv2.THRESH_BINARY)
cv2.imshow("Src - 127",dst)         # threshold = 127
thresh = 0                          # 定義閾值 = 0
ret, dst = cv2.threshold(src,thresh,maxval,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("Dst - Otsu",dst)        # Otsu
print(f"threshold = {ret}")

cv2.waitKey(0)
cv2.destroyAllWindows()                            






