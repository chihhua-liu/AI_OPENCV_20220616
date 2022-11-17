# ch9_9.py
import cv2
import numpy as np

thresh = 127                        # 定義閾值
maxval = 255                        # 定義像素最大值
src = np.random.randint(0,256,size=[3,5],dtype=np.uint8)
ret, dst = cv2.threshold(src,thresh,maxval,cv2.THRESH_TRUNC)
print(f"src =\n {src}")
print(f"threshold = {ret}")
print(f"dst =\n {dst}")






