# ch16_14.py
import cv2

src = cv2.imread("star.jpg")
cv2.imshow("src",src)
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)     # 影像轉成灰階
# 二值化處理影像
ret, dst_binary = cv2.threshold(src_gray,127,255,cv2.THRESH_BINARY)
# 找尋影像內的輪廓
contours, hierarchy = cv2.findContours(dst_binary,
                      cv2.RETR_LIST,
                      cv2.CHAIN_APPROX_SIMPLE)
# 凸包 -> 凸包缺陷
contour = contours[0]                               # 輪廓
hull = cv2.convexHull(contour,returnPoints = False) # 獲得凸包       
defects = cv2.convexityDefects(contour,hull)        # 獲得凸包缺陷
n = defects.shape[0]                                # 缺陷數量
print(f"缺陷數量 = {n}")
for i in range(n):
# s是startPoint, e是endPoint, f是farPoint, d是depth
    s, e, f, d = defects[i,0]
    start = tuple(contour[s][0])                    # 取得startPoint座標
    end = tuple(contour[e][0])                      # 取得endPoint座標
    far = tuple(contour[f][0])                      # 取得farPoint座標
    dst = cv2.line(src,start,end,[0,255,0],2)       # 凸包連線
    dst = cv2.circle(src,far,3,[0,0,255],-1)        # 繪製farPoint
cv2.imshow("dst",dst)           

cv2.waitKey(0)
cv2.destroyAllWindows()




