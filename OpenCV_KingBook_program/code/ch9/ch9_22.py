# ch9_22.py
import cv2
import numpy as np

img = cv2.imread("jk.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("JK Hung",img)

row, column = img.shape
x = np.zeros((row,column,8),dtype=np.uint8)
for i in range(8):
    x[:,:,i] = 2**i                             # 填上權重
result = np.zeros((row,column,8),dtype=np.uint8)
for i in range(8):
    result[:,:,i] = cv2.bitwise_and(img,x[:,:,i])
    mask = result[:,:,i] > 0                    # 影像邏輯值
    result[mask] = 255                          # True的位置填255
    cv2.imshow(str(i),result[:,:,i])            # 顯示影像
    
cv2.waitKey(0)
cv2.destroyAllWindows()




