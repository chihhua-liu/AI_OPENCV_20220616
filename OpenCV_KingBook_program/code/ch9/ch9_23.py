# ch9_23.py
import cv2
import numpy as np

jk = cv2.imread("jk.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("JK Hung",jk)                            # 顯示原始影像

row, column = jk.shape                              # 取得列高和欄寬
h7 = np.ones((row,column),dtype=np.uint8) * 254     # 建立像素值是254的影像
cv2.imshow("254",h7)                                # 顯示像素值是254的影像         
new_jk = cv2.bitwise_and(jk,h7)                     # 原始影像最低有效位元是 0               
cv2.imshow("New JK",new_jk)                         # 顯示新影像
    
cv2.waitKey(0)
cv2.destroyAllWindows()




