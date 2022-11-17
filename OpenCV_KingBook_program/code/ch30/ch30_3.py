# ch30_3.py
import cv2
import os
import glob
import time
import shutil

dstDir = "dstCar"
bmpDir = "bmpCar"
if os.path.isdir(bmpDir):                           # 檢查是否存在
# 因為bmpDir資料夾可能含資料, 所以使用shutil.rmtree()函數刪除
    shutil.rmtree(bmpDir)                           # 先刪除資料夾
    time.sleep(3)                                   # 休息讓系統處理              
os.mkdir(bmpDir)

allcars = dstDir + "/*.JPG"                         # 建立檔案模式
cars = glob.glob(allcars)                           # 獲得檔案名稱
#print(f"目前資料夾檔案名稱 = \n{cars}")            # 列印檔案名稱
# 拆解資料夾符號
for car in cars:
    carname = car.split("\\")                       # 將字串轉成串列
    #print(carname)
    car_img = cv2.cv2.imread(car,cv2.IMREAD_COLOR)  # 讀車子影像
    outname = carname[1].replace(".jpg", ".bmp")    # 將jpg改為bmp
    fullpath = bmpDir + "\\" + outname              # 完整檔名
    cv2.imwrite(fullpath, car_img)                  # 寫入資料夾
print("在 bmpCar 資瞭夾重新命名車輛副檔名成功")












