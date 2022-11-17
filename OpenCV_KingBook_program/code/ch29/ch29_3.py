# ch29_3.py
import cv2

# 建立對應名字的字典
faceNames = {"0":"Hung", "1":"Hung", "2":"Unistar", "3":"Unistar"}
model = cv2.face.LBPHFaceRecognizer_create()    # 建立人臉辨識物件
model.read("ch29_2\\model.yml")                 # 讀取人臉辨識數據模型
# 讀取要辨識的人臉
face = cv2.imread("ch29_2\\face.jpg",cv2.IMREAD_GRAYSCALE)
label,confidence = model.predict(face)          # 執行人臉辨識
print(f"Name       = {faceNames[str(label)]}")
print(f"Confidence = {confidence:6.2f}")


