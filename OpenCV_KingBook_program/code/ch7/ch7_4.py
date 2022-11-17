# ch7_4.py
import cv2


img = cv2.imread("antarctic.jpg")               # 使用影像當畫布
img[1:300,1:300] = (0,255,255)                  # 設定黃色底             

cv2.line(img,(1,1),(300,1),(255,0,0))           # 上方水平直線
cv2.line(img,(300,1),(300,300),(255,0,0))       # 右邊垂直直線
cv2.line(img,(300,300),(1,300),(255,0,0))       # 下邊水平直線
cv2.line(img,(1,300),(1,1),(255,0,0))           # 左邊垂直直線
for x in range(150, 300, 10):
    cv2.line(img,(x,1),(300,x-150),(255,0,0))
for y in range(150, 300, 10):
    cv2.line(img,(1,y),(y-150,300),(255,0,0))
    
cv2.imshow("My Draw",img)                       # 畫布顯示結果
cv2.waitKey(0)                                      
cv2.destroyAllWindows()                         # 刪除所有視窗






 
