# ch9_24.py
import cv2
import numpy as np

jk = cv2.imread("jk.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("JK Hung",jk)                            # 顯示原始影像

row, column = jk.shape                              # 取得列高和欄寬
h7 = np.ones((row,column),dtype=np.uint8) * 254     # 建立像素值是254的影像        
tmp_jk = cv2.bitwise_and(jk,h7)                     # 原始影像最低有效位元是 0
watermark = cv2.imread("copyright.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("Copy Right",watermark)                  # 顯示浮水印影像
ret, wm = cv2.threshold(watermark,0,1,cv2.THRESH_BINARY)
# 浮水印影像嵌入最低有效位元是 0的原始影像
new_jk = cv2.bitwise_or(tmp_jk, wm)                 
cv2.imshow("New JK",new_jk)                         # 顯示新影像
    
cv2.waitKey(0)
cv2.destroyAllWindows()




