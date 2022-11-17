# ch7_12.py
import cv2
import numpy as np

img = cv2.imread("antarctic.jpg")       # 使用影像當畫布
cy = int(img.shape[0] / 2)              # 中心點 y 座標
cx = int(img.shape[1] / 2)              # 中心點 x 座標
size = (200,100)                        # 橢圓的x,y軸長度
for i in range(0,15):
    angle = np.random.randint(0,361)    # 橢圓偏移的角度
    color = np.random.randint(0,256,size=3).tolist()    # 橢圓的隨機色彩
    cv2.ellipse(img,(cx,cy),size,angle,0,360,color,1)   # 繪製橢圓形
      
cv2.imshow("My Draw",img)               # 畫布顯示結果
cv2.waitKey(0)                                      
cv2.destroyAllWindows()                 # 刪除所有視窗






 
