# ch8_7.py
import cv2
import numpy as np

b = np.zeros((200,250,3),np.uint8)          # b影像
g = np.zeros((200,250,3),np.uint8)          # g影像
r = np.zeros((200,250,3),np.uint8)          # r影像
b[:,:,0] = 255                              # 設定藍色
g[:,:,1] = 255                              # 設定綠色
r[:,:,2] = 255                              # 設定紅色
cv2.imshow("B channel", b)                  # 顯示影像b
cv2.imshow("G channel", g)                  # 顯示影像g
cv2.imshow("R channel", r)                  # 顯示影像r

img1 = cv2.add(b,g)                         # b + g影像
cv2.imshow("B + G",img1)
img2 = cv2.add(g,r)                         # g + r影像
cv2.imshow("G + R",img2)
img3 = cv2.add(img1,r)                      # b + g + r影像
cv2.imshow("B + G + R",img3)

cv2.waitKey(0)                              # 等待
cv2.destroyAllWindows()                     # 刪除所有視窗








 
