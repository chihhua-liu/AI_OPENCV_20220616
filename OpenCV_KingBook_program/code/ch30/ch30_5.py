# ch30_5.py
# 標記檢查
import cv2
import os
import shutil
import time

dstDir = "plate-mark"
path = "Haar-Training-car-plate/training/positive/"

if os.path.isdir(dstDir):                           # 檢查是否存在
# 因為notCarDir資料夾可能含資料, 所以使用shutil.rmtree()函數刪除
    shutil.rmtree(dstDir)                           # 先刪除資料夾
    time.sleep(3)                                   # 休息讓系統處理              
os.mkdir(dstDir)

fn = open(path + 'info.txt', 'r')
row = fn.readline()                                 # 讀取info.txt
while row:
    msg = row.split(' ')                            # 分割每一列文字
    img = cv2.imread(path + msg[0])                 # 讀檔案
    n = int(msg[1])
    for i in range(n):
        x = int(msg[2 + i * 4])                     # 取得左上方 x 座標
        y = int(msg[3 + i * 4])                     # 取得左上方 y 座標
        w = int(msg[4 + i * 4])                     # 取得 width 寬度
        h = int(msg[5 + i * 4])                     # 取得 height 高度
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    imgname = (msg[0].split("/"))[-1]               # 使用-1是確定最右索引
    print(imgname)                                  # 輸出處理過程
    cv2.imwrite(dstDir + "\\" + imgname, img)       # 寫入資料夾
    row = fn.readline()
fn.close()
print("繪製車牌框完成")



        











