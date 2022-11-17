# ch7_17.py
import cv2

img = cv2.imread("antarctic.jpg")
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Antarctic',(120,120),font,3,(255,0,0),12)

cv2.imshow("Antarctic",img)                 
cv2.waitKey(0)                                      
cv2.destroyAllWindows()                     # 刪除所有視窗









 
