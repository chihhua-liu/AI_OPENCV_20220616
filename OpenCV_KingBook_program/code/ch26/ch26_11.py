# ch26_11.py
import cv2

capture = cv2.VideoCapture(0)       # 初始化攝影功能
while(capture.isOpened()):
    ret, frame = capture.read()     # 讀取設請鏡頭的影像
    cv2.imshow('Frame',frame)       # 顯示攝影鏡頭的影像
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)    # 寬度
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度
    c = cv2.waitKey(1)              # 等待時間 1 毫秒 ms
    if c == 27:                     # 按 Esc 键
        break
print(f"Frame 的寬度 = {width}")    # 輸出Frame 的寬度
print(f"Frame 的高度 = {height}")   # 輸出Frame 的高度
capture.release()                   # 關閉攝影功能
cv2.destroyAllWindows()









