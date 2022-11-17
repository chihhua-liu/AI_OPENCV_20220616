# ch8_1.py
import cv2
import numpy as np

src1 = np.random.randint(0,256,size=[3,3],dtype=np.uint8)
src2 = np.random.randint(0,256,size=[3,3],dtype=np.uint8)
res = cv2.add(src1,src2)
print(f"src1 = \n {src1}")
print(f"src2 = \n {src2}")
print(f"dst = \n {src1+src2}")












 
