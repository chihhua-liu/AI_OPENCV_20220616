# ch6_6.py
import cv2
import numpy as np
 
# 建立藍色blue底的彩色影像陣列
blue = np.zeros((2,3,3),np.uint8)   
blue[:,:,0] = 255                   # 填滿藍色
print(f"blue =\n{blue}")            # 列印影像陣列
# 列印修訂前的像素點
print(f"blue[0,1,2] = {blue[0,1,2]}")

blue[0,1,2] = 50                    # 修訂像素點的單一通道
print("修訂後")
# 列印修訂後的像素點
print(f"blue =\n{blue}")            # 列印影像陣列
print(f"blue[0,1,2] = {blue[0,1,2]}")






 

