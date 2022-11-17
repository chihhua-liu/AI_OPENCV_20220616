# ch7_26.py
import cv2
import numpy as np

def onChange(x):
     pass   
        
def OnMouseAction(event, x, y, flags, param):
    # color可以產生隨機色彩
    color = np.random.randint(0,high = 256,size=3).tolist()
    r = np.random.randint(10, 50)               # 隨機10-50半徑的圓
    if event == cv2.EVENT_LBUTTONDOWN:          # 按一下滑鼠左鍵
        cv2.circle(image,(x,y),r,color,thickness)   # 隨機的圓
    
thickness = -1                                  # 預設寬度是 0
height = 400                                    # 視窗高度
width = 600                                     # 視窗寬度
image = np.ones((height,width,3),np.uint8) * 255
cv2.namedWindow("Draw Circle")
cv2.setMouseCallback("Draw Circle",OnMouseAction)
cv2.createTrackbar('Thickness','Draw Circle',0,1,onChange)
while 1:
    cv2.imshow("Draw Circle",image)
    key = cv2.waitKey(100)                      # 0.1秒檢查一次
    num = cv2.getTrackbarPos('Thickness','Draw Circle')
    if num == 0:
        thickness = -1                          # 實心設定
    else:
        thickness = 3                           # 寬度是 3 
    if key == ord('Q') or key == ord('q'):      # Q或q則結束
        break
cv2.destroyAllWindows()                         # 刪除所有視窗













 
