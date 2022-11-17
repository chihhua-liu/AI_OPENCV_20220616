# ch7_5.py
import cv2
import numpy as np

img = np.ones((350,500,3),np.uint8) * 255       # 建立白色底的畫布
cv2.rectangle(img,(1,1),(300,300),(255,0,0))    # 繪製矩形
for x in range(150, 300, 10):
    cv2.line(img,(x,1),(300,x-150),(255,0,0))
for y in range(150, 300, 10):
    cv2.line(img,(1,y),(y-150,300),(255,0,0))
    
cv2.imshow("My Draw",img)                       # 畫布顯示結果
cv2.waitKey(0)                                      
cv2.destroyAllWindows()                         # 刪除所有視窗






 
