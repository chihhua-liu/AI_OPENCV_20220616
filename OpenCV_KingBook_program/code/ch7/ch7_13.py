# ch7_13.py
import cv2
import numpy as np

img1 = np.ones((200,300,3),np.uint8) * 255              # 畫布1
pts = np.array([[150,50],[250,100],[150,150],[50,100]]) # 頂點陣列
cv2.polylines(img1,[pts],True,(255,0,0),5)              # 繪製封閉式多邊形

img2 = np.ones((200,300,3),np.uint8) * 255              # 畫布2
cv2.polylines(img2,[pts],False,(0,0,255),3)             # 繪製開放式多邊形

cv2.imshow("isClosed_True",img1)                    # 畫布顯示封閉式多邊形
cv2.imshow("isClosed_False",img2)                   # 畫布顯示開放式多邊形
cv2.waitKey(0)                                      
cv2.destroyAllWindows()                             # 刪除所有視窗










 
