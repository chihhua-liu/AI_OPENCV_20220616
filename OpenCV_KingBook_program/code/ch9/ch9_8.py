# ch9_8.py
import cv2

src = cv2.imread("numbers.jpg")
thresh = 127                    # 閾值 = 10
maxval = 255                    # 二值化的極大值
ret, dst = cv2.threshold(src,thresh,maxval,cv2.THRESH_BINARY_INV)
cv2.imshow("Src",src)
cv2.imshow("Dst - 127",dst)     # threshold = 127
thresh = 10                     # 更改閾值 = 10
ret, dst = cv2.threshold(src,thresh,maxval,cv2.THRESH_BINARY_INV)
cv2.imshow("Dst - 10",dst)      # threshold = 10

cv2.waitKey(0)
cv2.destroyAllWindows()                            






