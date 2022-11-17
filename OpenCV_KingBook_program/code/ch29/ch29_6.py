# ch29_6.py
import cv2
import os
import glob
import numpy as np

total = 5                                           # 人臉取樣數
pictPath = r'C:\opencv\data\haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier(pictPath)      # 建立辨識檔案物件
if not os.path.exists("ch29_6"):                    # 如果不存在ch29_6資料夾
    os.mkdir("ch29_6")                              # 就建立ch29_6
name = input("請輸入英文名字 : ")
if os.path.exists("ch29_6\\" + name):
    print("此名字的人臉資料已經存在")
else:
    os.mkdir("ch29_6\\" + name)
    cap = cv2.VideoCapture(0)                       # 開啟攝影機
    num = 1                                         # 影像編號
    while(cap.isOpened()):                          # 攝影機有開啟就執行迴圈   
        ret, img = cap.read()                       # 讀取影像
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1,
                minNeighbors = 3, minSize=(20,20))
        for (x, y, w, h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  # 藍色框住人臉
        cv2.imshow("Photo", img)                    # 顯示影像在OpenCV視窗
        key = cv2.waitKey(200)
        if ret == True:                             # 讀取影像如果成功
            imageCrop = img[y:y+h,x:x+w]                      # 裁切
            imageResize = cv2.resize(imageCrop,(160,160))     # 重製大小
            faceName = "ch29_6\\" + name + "\\" + name + str(num) + ".jpg"
            cv2.imwrite(faceName, imageResize)      # 儲存人臉影像           
            if num >= total:                        # 拍指定人臉數後才終止               
                if num == total:
                    print(f"拍攝第 {num} 次人臉成功")
                break
            print(f"拍攝第 {num} 次人臉成功")
            num += 1
    cap.release()                                   # 關閉攝影機
    cv2.destroyAllWindows()
# 讀取人臉樣本和放入faces_db, 同時建立標籤與人名串列
nameList = []                                       # 員工姓名
faces_db = []                                       # 儲存所有人臉
labels = []                                         # 建立人臉標籤
index = 0                                           # 員工編號索引
dirs = os.listdir('ch29_6')                         # 取得所有資料夾及檔案
for d in dirs:                                      # d是所有員工人臉的資料夾
    if os.path.isdir('ch29_6\\' + d):               # 獲得資料夾
        faces = glob.glob('ch29_6\\' + d + '\\*.jpg')  # 資料夾中所有人臉
        for face in faces:                          # 讀取人臉
            img = cv2.imread(face, cv2.IMREAD_GRAYSCALE)
            faces_db.append(img)                    # 人臉存入串列
            labels.append(index)                    # 建立數值標籤
        nameList.append(d)                          # 將英文名字加入串列
        index += 1
print(f"標籤名稱 = {nameList}")
print(f"標籤序號 ={labels}")
# 儲存人名串列，可在未來辨識人臉時使用
f = open('ch29_6\\employee.txt', 'w')
f.write(','.join(nameList))
f.close()

print('建立人臉辨識資料庫')
model = cv2.face.LBPHFaceRecognizer_create()        # 建立LBPH人臉辨識物件
model.train(faces_db, np.array(labels))             # 訓練LBPH人臉辨識
model.save('ch29_6\\deepmind.yml')                  # 儲存LBPH訓練數據
print('人臉辨識資料庫完成')


