# ch26_1.py
import cv2

capture = cv2.VideoCapture(0)       # 初始化攝影功能
while(capture.isOpened()):
    ret, frame = capture.read()     # 讀取設請鏡頭的影像
    cv2.imshow('Frame',frame)       # 顯示攝影鏡頭的影像
    c = cv2.waitKey(1)              # 等待時間 1 毫秒 ms
    if c == 27:                     # 按 Esc 键, 結束
        break
capture.release()                   # 關閉攝影功能
cv2.destroyAllWindows()









