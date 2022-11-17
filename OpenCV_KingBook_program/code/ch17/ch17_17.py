# ch17_17.py
import cv2
import numpy as np

src = cv2.imread('hand.jpg')  
cv2.imshow("src",src)
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(src_gray,50,255,cv2.THRESH_BINARY)  
contours, hierarchy = cv2.findContours(binary,
                      cv2.RETR_EXTERNAL,
                      cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]  
mask = np.zeros(src_gray.shape,np.uint8)    # 建立遮罩
mask = cv2.drawContours(mask,[cnt],-1,(255,255,255),-1)
cv2.imshow("mask",mask)
# 在src_gray影像的mask遮罩區域找尋最大像素與最小像素值
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src_gray,mask=mask)
print(f"最小像素值 = {minVal}")
print(f"最小像素值座標 = {minLoc}")
print(f"最大像素值 = {maxVal}")
print(f"最大像素值座標 = {maxLoc}")
cv2.circle(src,minLoc,20,[0,255,0],3)       # 最小像素值用綠色圓
cv2.circle(src,maxLoc,20,[0,0,255],3)       # 最大像素值用紅色圓
# 建立遮罩未來可以顯示此感興趣的遮罩區域
mask1 = np.zeros(src.shape,np.uint8)        # 建立遮罩
mask1 = cv2.drawContours(mask1,[cnt],-1,(255,255,255),-1)
cv2.imshow("mask1",mask1)
dst = cv2.bitwise_and(src,mask1)            # 顯示感興趣區域
cv2.imshow("dst",dst)

cv2.waitKey()
cv2.destroyAllWindows()
