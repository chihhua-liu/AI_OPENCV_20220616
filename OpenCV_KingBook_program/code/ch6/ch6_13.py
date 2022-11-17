# ch6_13.py
import cv2

img = cv2.imread("jk.jpg")              # 彩色讀取
cv2.imshow("Hung Image", img)           # 顯示影像
face = img[30:220,80:250]               # ROI
cv2.imshow("Face", face)                # 顯示影像

cv2.waitKey(0)                                      
cv2.destroyAllWindows()                 # 刪除所有視窗






 
