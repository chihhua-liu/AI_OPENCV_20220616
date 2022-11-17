# ch26_9.py
import cv2

video = cv2.VideoCapture('iceocean.mov')    # 開啟影片檔案

while(video.isOpened()):
    ret, frame = video.read()               # 讀取影片檔案          
    if ret:
        cv2.imshow('frame',frame)           # 顯示影片
        c = cv2.waitKey(50)                 # 可以控制撥放速度
    else:
        break
    if c == 32:                             # 是否按 空白鍵 
        cv2.waitKey(0)                      # 等待按鍵發生
        continue
    if c == 27:                             # 按 Esc 键, 結束
        break

video.release()                             # 關閉輸出物件
cv2.destroyAllWindows()








