# ch8_18.py
import cv2
import numpy as np

src = cv2.imread("forest.jpg")                      # 讀取影像
key = np.random.randint(0,256,src.shape,np.uint8)   # 密鑰影像
print(src.shape)
cv2.imshow("forest",src)                            # 原始影像
cv2.imshow("key",key)                               # 密鑰影像

img_encry = cv2.bitwise_xor(src,key)                # 加密結果的影像
img_decry = cv2.bitwise_xor(key,img_encry)          # 解密結果的影像
cv2.imshow("encrytion",img_encry)                   # 加密結果影像
cv2.imshow("decrytion",img_decry)                   # 解密結果影像

cv2.waitKey()
cv2.destroyAllWindows()                             # 刪除所有視窗






