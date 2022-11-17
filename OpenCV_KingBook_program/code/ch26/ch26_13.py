# ch26_13.py
import cv2
       
capture = cv2.VideoCapture(0)       # 初始化攝影功能
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 設定寬度
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,960)   # 設定高度
while(capture.isOpened()):
    ret, frame = capture.read()     # 讀取設請鏡頭的影像
    cv2.imshow('Frame',frame)       # 顯示攝影鏡頭的影像
    c = cv2.waitKey(1)              # 等待時間 1 毫秒 ms
    if c == 27:                     # 按 Esc 键
        break
capture.release()                   # 關閉攝影功能
cv2.destroyAllWindows()








