# ch7_22.py
import cv2
import numpy as np
def OnMouseAction(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:          # 按一下滑鼠左鍵
        print(f"在x={x}, y={y}, 按一下滑鼠左鍵")
    elif event == cv2.EVENT_RBUTTONDOWN:        # 按一下滑鼠右鍵
        print(f"在x={x}, y={y}, 按一下滑鼠右鍵_")
    elif event == cv2.EVENT_MBUTTONDOWN:        # 按一下滑鼠中間鍵
        print(f"在x={x}, y={y}, 按一下滑鼠中間鍵")
    elif flags == cv2.EVENT_FLAG_LBUTTON:       # 按住滑鼠左鍵拖曳
        print(f"在x={x}, y={y}, 按住滑鼠左鍵拖曳")
    elif flags == cv2.EVENT_FLAG_RBUTTON:       # 按住滑鼠右鍵拖曳
        print(f"在x={x}, y={y}, 按住滑鼠右鍵拖曳")

image = np.ones((200,300,3),np.uint8) * 255
cv2.namedWindow("OpenCV Mouse Event")
cv2.setMouseCallback("OpenCV Mouse Event",OnMouseAction)
cv2.imshow("OpenCV Mouse Event",image)

cv2.waitKey(0)                                      
cv2.destroyAllWindows()                         # 刪除所有視窗













 
