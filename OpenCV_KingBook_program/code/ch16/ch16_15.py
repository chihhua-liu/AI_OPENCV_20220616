# ch16_15.py
import cv2

src = cv2.imread("heart1.jpg")
cv2.imshow("src",src)
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)     # 影像轉成灰階
# 二值化處理影像
ret, dst_binary = cv2.threshold(src_gray,127,255,cv2.THRESH_BINARY)
# 找尋影像內的輪廓
contours, hierarchy = cv2.findContours(dst_binary,
                      cv2.RETR_LIST,
                      cv2.CHAIN_APPROX_SIMPLE)
# 凸包
src1 = src.copy()                                   # 複製src影像
hull = cv2.convexHull(contours[0])                  # 獲得凸包頂點座標       
dst1 = cv2.polylines(src1, [hull], True, (0,255,0),2) # 將凸包連線
cv2.imshow("dst1",dst1)
isConvex = cv2.isContourConvex(hull)                # 是否凸形
print(f"凸包是凸形       = {isConvex}")
# 近似多邊形包圍
src2 = src.copy()                                   # 複製src影像
approx = cv2.approxPolyDP(contours[0], 10, True)    # epsilon=10
dst2 = cv2.polylines(src2,[approx],True,(0,255,0),2)  # 近似多邊形連線
cv2.imshow("dst2 - epsilon = 10",dst2)
isConvex = cv2.isContourConvex(approx)              # 是否凸形
print(f"近似多邊形是凸形 = {isConvex}")

cv2.waitKey(0)
cv2.destroyAllWindows()




