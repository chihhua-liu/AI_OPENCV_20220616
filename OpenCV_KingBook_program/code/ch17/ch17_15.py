# ch17_15.py
import cv2
import numpy as np

src = cv2.imread("simple.jpg")
cv2.imshow("src",src)
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)     # 影像轉成灰階
# 二值化處理影像
ret, dst_binary = cv2.threshold(src_gray,127,255,cv2.THRESH_BINARY)
# 找尋影像內的輪廓
contours, hierarchy = cv2.findContours(dst_binary,
                      cv2.RETR_LIST,
                      cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]                                   # 取得輪廓數據
mask1 = np.zeros(src_gray.shape,np.uint8)           # 建立畫布
dst1 = cv2.drawContours(mask1,[cnt],0,255,1)        # 繪製空心輪廓
points1 = cv2.findNonZero(dst1)
mask2 = np.zeros(src_gray.shape,np.uint8)           # 建立畫布
dst2 = cv2.drawContours(mask2,[cnt],0,255,-1)       # 繪製實心輪廓
points2 = cv2.findNonZero(dst2)
print(f"空心像素點長度 = {len(points1)},   實心像素點長度 = {len(points2)}")
print("空心像素點")
print(points1)
print("實心像素點")
print(points2)
cv2.imshow("dst1",dst1)
cv2.imshow("dst2",dst2)

cv2.waitKey(0)
cv2.destroyAllWindows()






