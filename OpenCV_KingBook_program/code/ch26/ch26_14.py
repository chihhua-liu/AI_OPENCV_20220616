# ch26_14.py
import cv2

video = cv2.VideoCapture('iceocean.mov')    # 開啟影片檔案
video_fps = video.get(cv2.CAP_PROP_FPS)     # 計算速度
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)   # 影片高度
counter = 1                                 # 幀數計數器
font = cv2.FONT_HERSHEY_SIMPLEX             # 字型
while(video.isOpened()):
    ret, frame = video.read()               # 讀取影片檔案          
    if ret:
        y = int(height - 50)                # Frames計數器位置
        cv2.putText(frame,'Frames  : ' + str(counter), (0, y),
                    font,1,(255,0,0),2)     # 顯示幀數
        seconds = round(counter / video_fps, 2)     # 計算秒數
        y = int(height - 10)                # Seconds計數器位置
        cv2.putText(frame,'Seconds : ' + str(seconds), (0, y),
                    font,1,(255,0,0),2)      # 顯示秒數
        cv2.imshow('myVideo',frame)         # 顯示影片
    else:
        break
    c = cv2.waitKey(50)                     # 可以控制撥放速度
    counter += 1                            # 幀數加 1
    if c == 27:                             # 按 Esc 键, 結束
        break

video.release()                             # 關閉輸出物件
cv2.destroyAllWindows()








