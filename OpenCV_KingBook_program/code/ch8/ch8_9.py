# ch8_9.py
import cv2
import numpy as np

src1 = np.ones((2,3),dtype=np.uint8) * 10           # 影像 src1
src2 = np.ones((2,3),dtype=np.uint8) * 50           # 影像 src2
alpha = 1
beta = 0.5
gamma = 5
print(f"src1 = \n {src1}")
print(f"src2 = \n {src2}")
dst = cv2.addWeighted(src1,alpha,src2,beta,gamma)   # 加權和
print(f"dst = \n {dst}")












 
