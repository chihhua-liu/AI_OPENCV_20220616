# ch29_1_1.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("ch29_1\\hung1.jpg",cv2.IMREAD_COLOR)    # 彩色讀取
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                # 轉RGB
plt.subplot(121)
plt.imshow(img)                                             # 顯示人臉
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      # 轉灰階
recognizer = cv2.face.LBPHFaceRecognizer_create()   # 建立人臉辨識物件
recognizer.train([gray], np.array([0]))             # 訓練人臉辨識
histogram = recognizer.getHistograms()[0][0]
axis_values = np.array([i for i in range(0, len(histogram))])
plt.subplot(122)
plt.bar(axis_values, histogram)
plt.show()





