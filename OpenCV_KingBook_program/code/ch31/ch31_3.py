# ch31_3.py
import cv2
import pytesseract

config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
pictPath = "haar_carplate.xml"                          # 哈爾特徵檔路徑
img = cv2.imread("testCar/cartest1.jpg")                # 讀辨識的影像
car_cascade = cv2.CascadeClassifier(pictPath)           # 讀哈爾特徵檔
# 執行辨識
plates = car_cascade.detectMultiScale(img, scaleFactor=1.05,
         minNeighbors=3, minSize=(20,20), maxSize=(155,50))  
if len(plates) > 0 :                                    # 有偵測到車牌
    for (x, y, w, h) in plates:                         # 標記車牌  
        carplate = img[y:y+h, x:x+w]                    # 車牌影像        
else:
    print("偵測車牌失敗")

cv2.imshow('Car', carplate)                             # 顯示所讀取的車輛
text = pytesseract.image_to_string(carplate,config=config)  # OCR辨識
print(f"車號是 : {text}")

cv2.waitKey(0)
cv2.destroyAllWindows()



        











