# ch29_2.py
import cv2
import numpy as np

face_db = [                                     # 人臉資料庫
            "ch29_2\\hung1.jpg",
            "ch29_2\\hung2.jpg",
            "ch29_2\\star1.jpg",
            "ch29_2\\star2.jpg"
          ]

faces = []                                      # 人臉空串列
for f in face_db:
    img = cv2.imread(f,cv2.IMREAD_GRAYSCALE)    # 讀取人臉資料庫
    faces.append(img)                           # 加入人臉空串列
# 建立標籤串列
labels = np.array([i for i in range(0, len(faces))])    
# 建立對應名字的字典            
model = cv2.face.LBPHFaceRecognizer_create()    # 建立人臉辨識物件
model.train(faces, np.array(labels))            # 訓練人臉辨識
model.save("ch29_2\\model.yml")                 # 儲存訓練的人臉數據
print("儲存訓練數據完成")









