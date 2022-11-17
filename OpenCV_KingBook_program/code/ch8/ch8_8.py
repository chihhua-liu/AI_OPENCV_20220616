# ch8_8.py
import cv2
import numpy as np

img1 = np.ones((4,5),dtype=np.uint8) * 8
img2 = np.ones((4,5),dtype=np.uint8) * 9 
mask = np.zeros((4,5),dtype=np.uint8)
mask[1:3,1:4] = 255
dst = np.random.randint(0,256,(4,5),np.uint8)
print("img1 = \n",img1)
print("img2 = \n",img2)
print("mask = \n",mask)
print("最初值 dst =\n",dst)
dst = cv2.add(img1,img2,mask=mask)
print("結果值 dst =\n",dst)




 
