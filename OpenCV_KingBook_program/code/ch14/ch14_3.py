# ch14_3.py
import cv2
import numpy as np

src1 = np.random.randint(256, size=(2,3),dtype = np.uint8) 
src2 = np.random.randint(256, size=(2,3),dtype = np.uint8)
dst = src1 + src2
print(f"src1 = \n{src1}")
print(f"src2 = \n{src2}")
print(f"dst = \n{dst}")




