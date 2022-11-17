# ch28_2.py
import cv2
import os

if not os.path.exists("ch28_2"):                # 如果不存在ch28_2資料夾
    os.mkdir("ch28_2")                          # 就建立ch28_2
name = input("請輸入英文名字 : ")
faceName = "ch28_2\\" + name + ".jpg"           # 人臉影像
facePhoto = "ch28_2\\" + name + "photo.jpg"     # 拍攝影像
pictPath = r'C:\opencv\data\haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier(pictPath)  # 建立辨識檔案物件
cap = cv2.VideoCapture(0)                       # 開啟攝影機
while(cap.isOpened()):                          # 攝影機有開啟就執行迴圈
    ret, img = cap.read()                       # 讀取影像
    cv2.imshow("Photo", img)                    # 顯示影像在OpenCV視窗
    if ret == True:                             # 讀取影像如果成功
        key = cv2.waitKey(200)                  # 0.2秒檢查一次
        if key == ord("a") or key == ord("A"):  # 如果按A或a
            cv2.imwrite(facePhoto, img)         # 將影像寫入facePhoto           
            break
cap.release()                                   # 關閉攝影機

img = cv2.imread(facePhoto)                     # 讀取影像facePhoto
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1,
        minNeighbors = 3, minSize=(20,20))

# 將人臉框起來
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  # 藍色框住人臉
    imageCrop = img[y:y+h,x:x+w]                    # 裁切
    imageResize = cv2.resize(imageCrop,(160,160))   # 重製大小
    cv2.imwrite(faceName, imageResize)              # 儲存人臉影像 
    
cv2.imshow("FaceRecognition", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

