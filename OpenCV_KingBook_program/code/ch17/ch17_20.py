# ch17_20.py
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
# 在src_gray影像的mask遮罩區域計算均值
mask = np.zeros(src_gray.shape,np.uint8)    # 建立遮罩
mask = cv2.drawContours(mask,[cnt],-1,(255,255,255),-1)
channels = cv2.mean(src, mask = mask)       # 計算遮罩的均值
print(channels)

cv2.waitKey(0)
cv2.destroyAllWindows()



