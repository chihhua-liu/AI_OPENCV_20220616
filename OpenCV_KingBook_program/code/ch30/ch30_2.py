# ch30_2.py
import os
import glob

dstDir = "dstCar"
allcars = dstDir + "/*.JPG"                 # 建立檔案模式
cars = glob.glob(allcars)                   # 獲得檔案名稱
print(f"目前資料夾檔案名稱 = \n{cars}")     # 列印檔案名稱
# 拆解資料夾符號
for car in cars:
    carname = car.split("\\")               # 將字串轉成串列
    print(carname)







