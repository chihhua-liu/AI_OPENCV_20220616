# ch8_11.py
import cv2
import numpy as np

src1 = np.random.randint(0,255,(3,5),dtype=np.uint8)
src2 = np.zeros((3,5),dtype=np.uint8)
src2[0:2,0:2] = 255
dst = cv2.bitwise_and(src1,src2)
print(f"src1 = \n {src1}")
print(f"src2 = \n {src2}")
print(f"dst = \n {dst}")




