# ch16_12_1.py
import cv2

src = cv2.imread("hand1.jpg")
cv2.imshow("src",src)
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)     # 影像轉成灰階
# 二值化處理影像
ret, dst_binary = cv2.threshold(src_gray,127,255,cv2.THRESH_BINARY)
# 找尋影像內的輪廓
contours, hierarchy = cv2.findContours(dst_binary,
                      cv2.RETR_LIST,
                      cv2.CHAIN_APPROX_SIMPLE)
# 凸包
hull = cv2.convexHull(contours[0])                  # 獲得凸包頂點座標       
dst = cv2.polylines(src, [hull], True, (0,255,0),2) # 將凸包連線
cv2.imshow("dst",dst)
convex_area = cv2.contourArea(hull)                 # 凸包面積
print(f"凸包面積 = {convex_area}")

cv2.waitKey(0)
cv2.destroyAllWindows()




