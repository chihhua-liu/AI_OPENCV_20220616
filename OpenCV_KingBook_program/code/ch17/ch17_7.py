# ch17_7.py
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
cnt = contours[0]                                   # 建立輪廓變數
left = tuple(cnt[cnt[:,:,0].argmin()][0])         # left
right = tuple(cnt[cnt[:,:,0].argmax()][0])        # right
top = tuple(cnt[cnt[:,:,1].argmin()][0])          # top
bottom = tuple(cnt[cnt[:,:,1].argmax()][0])       # bottom
print(f"最左點 = {left}")
print(f"最右點 = {right}")
print(f"最上點 = {top}")
print(f"最下點 = {bottom}")
dst = cv2.circle(src,left,5,[0,255,0],-1)
dst = cv2.circle(src,right,5,[0,255,0],-1)
dst = cv2.circle(src,top,5,[0,255,255],-1)
dst = cv2.circle(src,bottom,5,[0,255,255],-1)
cv2.imshow("dst",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()














