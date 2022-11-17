# ch9_25.py
import cv2
import numpy as np

jk = cv2.imread("jk.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("JK Hung",jk)                            # 顯示原始影像

row, column = jk.shape                              # 取得列高和欄寬
h7 = np.ones((row,column),dtype=np.uint8) * 254     # 建立像素值是254的影像
tmp_jk = cv2.bitwise_and(jk,h7)                     # 原始影像最低有效位元是 0

watermark = cv2.imread("copyright.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("original watermark",watermark)          # 顯示浮水印影像
ret, wm = cv2.threshold(watermark,0,1,cv2.THRESH_BINARY)

new_jk = cv2.bitwise_or(tmp_jk, wm)                 # 浮水印影像嵌入原始影像
cv2.imshow("New JK",new_jk)                         # 顯示新影像
# 擷取浮水印
h0 = np.ones((row,column),dtype=np.uint8)
wm = cv2.bitwise_and(new_jk, h0)
ret, dst = cv2.threshold(wm,0,255,cv2.THRESH_BINARY)
cv2.imshow("result Watermark",dst)                  # 顯示浮水印

cv2.waitKey(0)
cv2.destroyAllWindows()




