# ch29_7.py
import cv2

pictPath = r'C:\opencv\data\haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier(pictPath)      # 建立辨識物件

model = cv2.face.LBPHFaceRecognizer_create()
model.read('ch29_6\\deepmind.yml')                  # 讀取已訓練模型
f = open('ch29_6\\employee.txt', 'r')               # 開啟姓名標籤
names = f.readline().split(',')                     # 將姓名存於串列

cap = cv2.VideoCapture(0)
while(cap.isOpened()):                              # 如果開啟攝影機成功
    ret, img = cap.read()                           # 讀取影像
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1,
                minNeighbors = 3, minSize=(20,20))
    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  # 藍色框住人臉
    cv2.imshow("Face", img)                         # 顯示影像
    k = cv2.waitKey(200)                            # 0.2秒讀鍵盤一次
    if ret == True:       
        if k == ord("a") or k == ord("A"):          # 按 a 或 A 鍵
            imageCrop = img[y:y+h,x:x+w]                    # 裁切
            imageResize = cv2.resize(imageCrop,(160,160))   # 重製大小
            cv2.imwrite("ch29_6\\face.jpg", imageResize)    # 將測試人臉存檔
            break
cap.release()                                       # 關閉攝影機
cv2.destroyAllWindows()
# 讀取員工人臉
gray = cv2.imread("ch29_6\\face.jpg", cv2.IMREAD_GRAYSCALE)
val = model.predict(gray)
if val[1] < 50:                                     #人臉辨識成功
    print(f"歡迎Deepmind員工 {names[val[0]]} 登入")
    print(f"匹配值是 {val[1]:6.2f}")
else:
    print("對不起你不是員工, 請洽人事部門")

  


