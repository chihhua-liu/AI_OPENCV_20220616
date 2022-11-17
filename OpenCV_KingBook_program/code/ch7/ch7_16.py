# ch7_16.py
import cv2
import numpy as np

img = np.ones((300,600,3),np.uint8) * 255   # 畫布
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Python',(120,120),font,3,(255,0,0),12)
cv2.putText(img,'Python',(120,180),font,3,(0,255,0),12,
            cv2.LINE_8,True)

cv2.imshow("Python",img)                    # 畫布顯示文字
cv2.waitKey(0)                                      
cv2.destroyAllWindows()                     # 刪除所有視窗










 
