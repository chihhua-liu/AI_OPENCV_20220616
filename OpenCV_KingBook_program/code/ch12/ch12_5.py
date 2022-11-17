# ch12_5.py
import cv2
import numpy as np

src = np.zeros((7,7),np.uint8) 
src[2:5,2:5] = 1                    # 建立前景影像
kernel = np.ones((3,3),np.uint8)    # 建立內核
dst = cv2.dilate(src, kernel)       # 膨脹操作
print(f"src = \n {src}")
print(f"kernel = \n {kernel}")
print(f"Dilation = \n {dst}")





