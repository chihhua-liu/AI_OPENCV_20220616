# ch27_17.py
import cv2

pictPath = r'C:\opencv\data\haarcascade_russian_plate_number.xml'
car_cascade = cv2.CascadeClassifier(pictPath)       # 建立辨識物件
img = cv2.imread("car.jpg")                         # 讀取影像
plates = car_cascade.detectMultiScale(img, scaleFactor=1.1,
        minNeighbors = 3, minSize=(20,20))
# 將車牌框起來, 由於有可能找到好幾個臉所以用迴圈繪出來
for (x,y,w,h) in plates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  # 藍色框住車牌
cv2.imshow("Car Plate", img)                        # 顯示影像

cv2.waitKey(0)
cv2.destroyAllWindows()




 
