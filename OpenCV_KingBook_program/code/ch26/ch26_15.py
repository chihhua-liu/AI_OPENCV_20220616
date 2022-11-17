# ch26_15.py
import cv2

video = cv2.VideoCapture('iceocean.mov')    # 開啟影片檔案
video_fps = video.get(cv2.CAP_PROP_FPS)     # 計算速度
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))    # 寬度
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度
# 建立裁剪影片物件
fourcc = cv2.VideoWriter_fourcc(*'I420')    # 編碼
new_video = cv2.VideoWriter('out26_15.avi', fourcc,
                            video_fps, (width, height))
counter = video_fps * 5                     # 影片長度
while(video.isOpened() and counter >= 0):
    ret, frame = video.read()               # 讀取影片檔案          
    if ret:
        new_video.write(frame)              # 寫入新影片
        counter -= 1                        # 幀數減 1

video.release()                             # 關閉輸出物件
cv2.destroyAllWindows()








