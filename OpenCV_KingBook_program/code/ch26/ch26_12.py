# ch26_12.py
import cv2

video = cv2.VideoCapture('iceocean.mov')   # 開啟影片檔案
while(video.isOpened()):
    ret, frame = video.read()              # 讀取影片檔案
    cv2.imshow('Frame',frame)              # 顯示影像
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)         # 寬度
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)       # 高度
    video_fps = video.get(cv2.CAP_PROP_FPS)             # 速度
    video_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)  # 幀數
    c = cv2.waitKey(50)                    # 等待時間
    if c == 27:                            # 按 Esc 键
        break
print(f"Video 的寬度    = {width}")        # 輸出 Video 的寬度
print(f"Video 的高度    = {height}")       # 輸出 Video 的高度
print(f"Video 的速度    = {video_fps}")    # 輸出 Video 的速度
print(f"Video 的幀數    = {video_frames}") # 輸出 Video 的幀數
video.release()                            # 關閉攝影功能
cv2.destroyAllWindows()









