# ch9_19.py
import cv2
import numpy as np

thresh = 0                          # 定義閾值
maxval = 255                        # 定義像素最大值
src = np.ones((3,4),dtype=np.uint8) * 120   # 設定陣列是 120
src[0:2,0:2]=108                    # 設定陣列區間為 0
ret, dst = cv2.threshold(src,thresh,maxval,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(f"src =\n {src}")
print(f"threshold = {ret}")
print(f"dst =\n {dst}")               






