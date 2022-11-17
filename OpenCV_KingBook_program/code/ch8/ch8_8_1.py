# ch8_8_1.py
import cv2
import numpy as np

img1 = np.zeros((200,300,3),np.uint8)       # 建立img1影像
img1[:,:,1] = 255
cv2.imshow("img1", img1)                    # 顯示影像img1
img2 = np.zeros((200,300,3),np.uint8)       # 建立img2影像
img2[:,:,2] = 255
cv2.imshow("img2", img2)                    # 顯示影像img2
m = np.zeros((200,300,1),np.uint8)          # 建立mask(m)影像
m[50:150,100:200,:] = 255                   # 建立 ROI
cv2.imshow("mask", m)                       # 顯示影像m

img3 = cv2.add(img1,img2)                   # 不含mask的影像相加
cv2.imshow("img1 + img2",img3)
img4 = cv2.add(img1,img2,mask=m)            # 含mask的影像相加
cv2.imshow("img1 + img2 + mask",img4)

cv2.waitKey(0)                              
cv2.destroyAllWindows()                     # 刪除所有視窗








 
