# ch27_10.py
import cv2

pictPath = r'C:\opencv\data\haarcascade_upperbody.xml'
body_cascade = cv2.CascadeClassifier(pictPath)      # 建立辨識物件
img = cv2.imread("people1.jpg")                     # 讀取影像
bodies = body_cascade.detectMultiScale(img, scaleFactor=1.1,
         minNeighbors = 9, minSize=(20,20))
# 標註身體
for (x,y,w,h) in bodies:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  # 藍色框住身體
cv2.imshow("Face", img)                             # 顯示影像

cv2.waitKey(0)
cv2.destroyAllWindows()




 
