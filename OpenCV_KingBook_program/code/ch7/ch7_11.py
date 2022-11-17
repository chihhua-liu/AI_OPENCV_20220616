# ch7_11.py
import cv2

img = cv2.imread("antarctic.jpg")       # 使用影像當畫布
cy = int(img.shape[0] / 2)              # 中心點 y 座標
cx = int(img.shape[1] / 2)              # 中心點 x 座標
red = (0, 0, 255)                       # 設定紅色      
yellow = (0,255,255)                    # 設定黃色
blue = (255,0,0)                        # 設定藍色
size = (200,100)
angle = 0
cv2.ellipse(img,(cx,cy),size,angle,0,360,red,1)     # 繪製橢圓形
angle = 45
cv2.ellipse(img,(cx,cy),size,angle,0,360,yellow,5)  # 繪製橢圓形
cv2.ellipse(img,(cx,cy),size,angle,45,135,blue,3)   # 繪製橢圓弧
      
cv2.imshow("My Draw",img)               # 畫布顯示結果
cv2.waitKey(0)                                      
cv2.destroyAllWindows()                 # 刪除所有視窗






 
