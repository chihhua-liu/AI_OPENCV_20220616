# ch17_8.py
import cv2

src = cv2.imread("explode1.jpg")
cv2.imshow("src",src)
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)     # 影像轉成灰階
# 二值化處理影像
ret, dst_binary = cv2.threshold(src_gray,127,255,cv2.THRESH_BINARY)
# 找尋影像內的輪廓
contours, hierarchy = cv2.findContours(dst_binary,
                      cv2.RETR_LIST,
                      cv2.CHAIN_APPROX_SIMPLE)  
dst = cv2.drawContours(src,contours,-1,(0,255,0),3) # 繪製輪廓
con_area = cv2.contourArea(contours[0])             # 輪廓面積
x, y, w, h = cv2.boundingRect(contours[0])          # 建構矩形
dst = cv2.rectangle(src,(x, y),(x+w, y+h),(0,255,255),2)
cv2.imshow("dst",dst)
square_area = w * h                                 # 計算矩形面積
extent = con_area / square_area                     # 計算Extent
print(f"Extent = {extent}")

cv2.waitKey(0)
cv2.destroyAllWindows()






