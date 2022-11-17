# ch30_1.py
import cv2
import os
import glob
import time
import shutil

srcDir = "srcCar"
dstDir = "dstCar"
width = 320
height = 240
if os.path.isdir(dstDir):                           # 檢查是否存在
# 因為dstCar資料夾可能含資料, 所以使用shutil.rmtree()函數刪除
    shutil.rmtree(dstDir)                           # 先刪除資料夾
    time.sleep(3)                                   # 休息讓系統處理              
os.mkdir(dstDir)                                    # 建立資料夾
# 取得資料夾底下所有車子影像名稱
cars = glob.glob(srcDir + "/*.jpg")                 
print(f"執行 {srcDir} 資料夾內尺寸的轉換 ... ")
for index, car in enumerate(cars, 1):               # 從1開始
    img_car = cv2.imread(car,cv2.IMREAD_COLOR)      # 讀車子影像
    img_car_resize = cv2.resize(img_car, (width, height))
    car_name = "car" + str(index) + ".jpg"          # 車子影像命名
    fullpath = dstDir + "\\" + car_name             # 完成路徑
    cv2.imwrite(fullpath, img_car_resize)           # 寫入車子影像
print(f"儲存 {dstDir} 資料夾內尺寸的轉換 ... ")
    






