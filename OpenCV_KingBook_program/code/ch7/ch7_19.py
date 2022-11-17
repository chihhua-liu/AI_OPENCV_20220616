# ch7_19.py
import cv2
import numpy as np
from random import *
import time

width = 640                             # 反彈球畫布寬度
height = 480                            # 反彈球畫布高度
r = 15                                  # 反彈球半徑
speed = 0.01                            # 反彈球移動速度
x = 50                                  # 反彈球的最初 x 位置 
y = 50                                  # 反彈球的最初 y 位置
x_step = 5                              # 反彈球移動 x 步伐
y_step = 5                              # 反彈球移動 y 步伐

while cv2.waitKey(1) == -1:
    if x > width - r or x < r:          # 反彈球超出畫布右邊界或是左邊界
        x_step = -x_step
    if y > height - r or y < r:         # 反彈球超出畫布下邊界或是上邊界
        y_step = -y_step
    x += x_step                         # 新的反彈球 x 位置
    y += y_step                         # 新的反彈球 y 位置
    img = np.ones((height, width, 3), np.uint8) * 255
    cv2.circle(img,(x,y),r,(255,0,0),-1)    # 繪製反彈球
    cv2.imshow("Bouncing Ball",img)                   
    time.sleep(speed)                       # 依speed設定休息
                                       
cv2.destroyAllWindows()                     # 刪除所有視窗










 
