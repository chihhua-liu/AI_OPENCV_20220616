# ch8_3_1.py
import cv2
import numpy as np

value = 20                                  # 亮度調整值
img = cv2.imread("jk.jpg")                  # 彩色讀取
coff = np.ones(img.shape,dtype=np.uint8) * value

res = cv2.add(img, coff)                    # 調整亮度結果
cv2.imshow("MyPicture1", img)               # 顯示影像img
cv2.imshow("MyPicture2", res)               # 顯示影像res

cv2.waitKey(0)                              # 等待
cv2.destroyAllWindows()                     # 刪除所有視窗












 
