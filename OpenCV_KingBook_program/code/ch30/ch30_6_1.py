# ch30_6_1.py
import cv2

pictPath = "haar_carplate.xml"                          # 哈爾特徵檔路徑
img = cv2.imread("testCar/cartest2.jpg")                # 讀辨識的影像
car_cascade = cv2.CascadeClassifier(pictPath)           # 讀哈爾特徵檔
# 執行辨識
plates = car_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=3,
         minSize=(20,20),maxSize=(155,50))  
if len(plates) > 0 :                                    # 有偵測到車牌
    for (x, y, w, h) in plates:                         # 標記車牌  
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        print(plates)
else:
    print("偵測車牌失敗")

cv2.imshow('Car', img)                                  # 顯示所讀取的車輛
cv2.waitKey(0)
cv2.destroyAllWindows()



        











