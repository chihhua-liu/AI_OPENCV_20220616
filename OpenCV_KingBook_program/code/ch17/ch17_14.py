# ch17_14.py
import cv2
import numpy as np

height = 3                                      # 矩陣高度 
width = 5                                       # 矩陣寬度
img = np.random.randint(2,size=(height,width))  # 建立0, 1矩陣
print(f"矩陣內容 = \n{img}")
loc_img = cv2.findNonZero(img)                  # 獲得非0元素座標
print(f"非0元素的座標 \n{loc_img}")













