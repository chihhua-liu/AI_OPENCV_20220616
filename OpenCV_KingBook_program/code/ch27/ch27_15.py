# ch27_15.py
import cv2

pictPath = r'C:\opencv\data\haarcascade_frontalcatface.xml'
cat_cascade = cv2.CascadeClassifier(pictPath)           # 建立辨識物件
img = cv2.imread("cat1.jpg")                            # 讀取影像
faces = cat_cascade.detectMultiScale(img, scaleFactor=1.1,
        minNeighbors = 3, minSize=(20,20))
# 將貓臉框起來, 由於有可能找到好幾個臉所以用迴圈繪出來
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)      # 藍色框住貓臉
cv2.imshow("Face", img)                                 # 顯示影像

cv2.waitKey(0)
cv2.destroyAllWindows()




 
