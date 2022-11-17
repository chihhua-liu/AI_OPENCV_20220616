# ch27_13.py
import cv2

pictPath1 = r'C:\opencv\data\haarcascade_frontalface_default.xml'
pictPath2 = r'C:\opencv\data\haarcascade_lefteye_2splits.xml'

face_cascade = cv2.CascadeClassifier(pictPath1)         # 建立人臉物件
img = cv2.imread("jk.jpg")                              # 讀取影像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 偵測人臉
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1,
        minNeighbors = 3, minSize=(20,20))
# 偵測左眼
eyes_cascade = cv2.CascadeClassifier(pictPath2)         # 建立左眼物件
eyes = eyes_cascade.detectMultiScale(img, scaleFactor=1.1,
        minNeighbors = 7, minSize=(20,20))
# 將人臉框起來, 由於有可能找到好幾個臉所以用迴圈繪出來
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)      # 藍色框住人臉
# 將左眼框起來, 由於有可能找到好幾個眼睛所以用迴圈繪出來
for (x,y,w,h) in eyes:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)      # 綠色框住眼睛    
cv2.imshow("Face", img)                                 # 顯示影像

cv2.waitKey(0)
cv2.destroyAllWindows()






 
