# ch17_10.py
import cv2
import numpy as np

src = cv2.imread("star1.jpg")
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
ed = np.sqrt(4 * con_area / np.pi)                  # 計算等效面積
print(f"等效面積 = {ed}")
dst = cv2.circle(src,(260,110),int(ed/2),(0,255,0),3)   # 繪製圓
cv2.imshow("dst",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()






