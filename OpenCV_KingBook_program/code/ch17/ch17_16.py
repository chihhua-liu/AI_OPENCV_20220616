# ch17_16.py
import cv2
import numpy as np

height = 3                                          # 矩陣高度 
width = 5                                           # 矩陣寬度
img = np.random.randint(256,size=(height,width))    # 建立矩陣
print(f"矩陣內容 = \n{img}")
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img)
print(f"最小值 = {minVal},  位置 = {minLoc}")       # 最小值與其位置
print(f"最大值 = {maxVal},  位置 = {maxLoc}")       # 最大值與其位置










