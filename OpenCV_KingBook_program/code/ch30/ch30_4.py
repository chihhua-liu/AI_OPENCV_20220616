# ch30_4.py
import cv2
import os
import glob
import shutil
import time

srcDir = "notCar"
dstDir = "notCarGray"
width = 500                                         # 負樣本寬     
height = 400                                        # 負樣本高
if os.path.isdir(dstDir):                           # 檢查是否存在
# 因為notCarDir資料夾可能含資料, 所以使用shutil.rmtree()函數刪除
    shutil.rmtree(dstDir)                           # 先刪除資料夾
    time.sleep(3)                                   # 休息讓系統處理              
os.mkdir(dstDir)

allcars = srcDir + "/*.JPG"                         # 建立檔案模式
cars = glob.glob(allcars)                           # 獲得檔案名稱
for index, car in enumerate(cars, 1):
    img = cv2.imread(car,cv2.IMREAD_GRAYSCALE)      # 灰階讀車子影像
    img_resize = cv2.resize(img, (width, height))   # 調整負樣本影像
    imgname =  "notcar" + str(index)
    fullpath = dstDir + "\\" + imgname + ".jpg"
    cv2.imwrite(fullpath, img_resize)
print("在 notCar 資瞭夾將影像轉為灰階成功,同時存入notCarGray資料夾")












