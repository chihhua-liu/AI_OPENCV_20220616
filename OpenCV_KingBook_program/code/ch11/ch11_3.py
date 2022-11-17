# ch11_3.py
import cv2
import numpy as np

src = np.ones((3,3), np.float32) * 150
src[1,1] = 20
print(f"src = \n {src}")
dst = cv2.medianBlur(src, 3)
print(f"dst = \n {dst}")
