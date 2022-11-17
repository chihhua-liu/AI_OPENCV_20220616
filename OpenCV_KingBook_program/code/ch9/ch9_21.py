# ch9_21.py
import cv2

src = cv2.imread("school.jpg",cv2.IMREAD_GRAYSCALE)     # 灰階讀取
thresh = 127                                            # 閾值
maxval = 255                                            # 定義像素最大值
ret,dst = cv2.threshold(src,thresh,maxval,cv2.THRESH_BINARY)    # 二值化處理
# 自適應閾值計算方法為ADAPTIVE_THRESH_MEAN_C
dst_mean = cv2.adaptiveThreshold(src,maxval,cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY,3,5)
# 自適應閾值計算方法為ADAPTIVE_THRESH_GAUSSIAN_C 
dst_gauss = cv2.adaptiveThreshold(src,maxval,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,3,5)
cv2.imshow("src",src)                                   # 顯示原始影像
cv2.imshow("THRESH_BINARY",dst)                         # 顯示二值化處理影像
cv2.imshow("ADAPTIVE_THRESH_MEAN_C",dst_mean)           # 顯示自適應閾值結果
cv2.imshow("ADAPTIVE_THRESH_GAUSSIAN_C",dst_gauss)      # 顯示自適應閾值結果

cv2.waitKey(0)
cv2.destroyAllWindows()                          






