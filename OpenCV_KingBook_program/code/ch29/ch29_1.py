# ch29_1.py
import cv2
import numpy as np

face_db = []                                        # 建立空串列
face_db.append(cv2.imread("ch29_1\\hung1.jpg",cv2.IMREAD_GRAYSCALE))
face_db.append(cv2.imread("ch29_1\\hung2.jpg",cv2.IMREAD_GRAYSCALE))
face_db.append(cv2.imread("ch29_1\\star1.jpg",cv2.IMREAD_GRAYSCALE))
face_db.append(cv2.imread("ch29_1\\star2.jpg",cv2.IMREAD_GRAYSCALE))

labels = [0,0,1,1]                                  # 建立標籤串列
faceNames = {"0":"Hung", "1":"Unistar"}             # 建立對應名字的字典

recognizer = cv2.face.LBPHFaceRecognizer_create()   # 建立人臉辨識物件
recognizer.train(face_db, np.array(labels))         # 訓練人臉辨識
# 讀取要辨識的人臉
face = cv2.imread("ch29_1\\face.jpg",cv2.IMREAD_GRAYSCALE)
label,confidence = recognizer.predict(face)         # 執行人臉辨識
print(f"Name       = {faceNames[str(label)]}")
print(f"Confidence = {confidence:6.2f}")



