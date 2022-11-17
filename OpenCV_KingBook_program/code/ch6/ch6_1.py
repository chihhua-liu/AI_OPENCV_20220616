# ch6_1.py
import cv2
import numpy as np
 
# 建立GRAY影像陣列
image = np.zeros((5, 12), np.uint8)
print(f"修改前 image=\n{image}")        # 顯示修改前GRAY影像
print(f"image[1,4] = {image[1, 4]}")    # 列出特定像素點的內容

image[1,4] = 255                        # 修改像素點的內容
print(f"修改後 image=\n{image}")        # 顯示修改後的GRAY影像
print(f"image[1,4] = {image[1, 4]}")    # 列出特定像素點的內容






 

