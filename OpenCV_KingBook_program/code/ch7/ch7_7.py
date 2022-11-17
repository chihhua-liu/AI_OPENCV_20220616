# ch7_7.py
import cv2

img = cv2.imread("antarctic.jpg")       # 使用影像當畫布
cy = int(img.shape[0] / 2)              # 中心點 y 座標
cx = int(img.shape[1] / 2)              # 中心點 x 座標
red = (0, 0, 255)                       # 設定紅色
yellow = (0,255,255)                    # 設定黃色
cv2.circle(img,(cx,cy),30,red,-1)       # 繪製實心圓形
for r in range(40, 200, 20):            # 繪製系列空心圓形
    cv2.circle(img,(cx,cy),r,yellow,2)
    
cv2.imshow("My Draw",img)               # 畫布顯示結果
cv2.waitKey(0)                                      
cv2.destroyAllWindows()                 # 刪除所有視窗






 
