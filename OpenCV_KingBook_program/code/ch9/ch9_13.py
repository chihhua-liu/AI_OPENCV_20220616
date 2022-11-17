# ch9_13.py
import cv2

thresh = 127                        # 定義閾值
maxval = 255                        # 定義像素最大值
src = cv2.imread("jk.jpg",cv2.IMREAD_GRAYSCALE)
ret, dst = cv2.threshold(src,thresh,maxval,cv2.THRESH_TOZERO)
cv2.imshow("Src",src)
cv2.imshow("Dst - 127",dst)         # threshold = 127
thresh = 80                         # 修訂所定義的閾值
ret, dst = cv2.threshold(src,thresh,maxval,cv2.THRESH_TOZERO)
cv2.imshow("Dst - 80",dst)          # threshold = 80

cv2.waitKey(0)
cv2.destroyAllWindows()                            






