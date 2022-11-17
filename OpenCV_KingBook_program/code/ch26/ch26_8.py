# ch26_8.py
import cv2

video = cv2.VideoCapture('iceocean2.mov')   # 開啟影片檔案

while(video.isOpened()):
    ret, frame = video.read()               # 讀取影片檔案          
    if ret==True:
        cv2.imshow('frame',frame)           # 顯示彩色影片
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray_frame',gray_frame) # 顯示灰階影片
    else:
        break
    c = cv2.waitKey(50)                 # 可以控制撥放速度
    if c == 27:                         # 按 Esc 键, 結束
            break

video.release()                             # 關閉輸出物件
cv2.destroyAllWindows()








