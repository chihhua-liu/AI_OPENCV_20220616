# ch7_23.py
import cv2
import numpy as np

def OnMouseAction(event, x, y, flags, param):
    # color可以產生隨機色彩
    color = np.random.randint(0,high = 256,size=3).tolist()
    r = np.random.randint(10, 50)               # 隨機10-50半徑的圓
    if event == cv2.EVENT_LBUTTONDOWN:          # 按一下滑鼠左鍵       
        cv2.circle(image,(x,y),r,color,-1)      # 隨機的實心圓
    elif event == cv2.EVENT_RBUTTONDOWN:        # 按一下滑鼠右鍵         
         cv2.circle(image,(x,y),r,color,3)      # 隨機的空心圓

height = 400                                    # 視窗高度
width = 600                                     # 視窗寬度
image = np.ones((height,width,3),np.uint8) * 255
cv2.namedWindow("Draw Circle")
cv2.setMouseCallback("Draw Circle",OnMouseAction)
while 1:
    cv2.imshow("Draw Circle",image)
    key = cv2.waitKey(100)                      # 0.1秒檢查一次
    if key == ord('Q') or key == ord('q'):      # Q或q則結束
        break
   
cv2.destroyAllWindows()                         # 刪除所有視窗













 
