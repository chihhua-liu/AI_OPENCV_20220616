# ch26_5.py
import cv2

capture = cv2.VideoCapture(0)               # 初始化攝影功能
fourcc = cv2.VideoWriter_fourcc(*'XVID')    # MPEG-4
# 建立輸出物件
video_out = cv2.VideoWriter('out26_5.avi',fourcc, 20.0, (640,480))
while(capture.isOpened()):
    ret, frame = capture.read()                 
    if ret:
        video_out.write(frame)              # 寫入影片物件
        cv2.imshow('frame',frame)           # 顯示攝影鏡頭的影像
    c = cv2.waitKey(1)                      # 等待時間 1 毫秒 ms
    if c == 27:                             # 按 Esc 键, 結束
        break
capture.release()                           # 關閉攝影功能
video_out.release()                         # 關閉輸出物件
cv2.destroyAllWindows()








