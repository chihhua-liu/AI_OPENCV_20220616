# ch17_18.py
import cv2

src = cv2.imread('forest.png')  
cv2.imshow("src",src)
channels = cv2.mean(src)            # 計算均值
print(channels)

cv2.waitKey(0)
cv2.destroyAllWindows()



