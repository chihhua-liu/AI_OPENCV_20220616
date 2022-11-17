# ch25_14.py
import cv2
import numpy as np

# 下載數據
with np.load('knn_digit.npz') as data:
    train = data['train']
    train_labels = data['train_labels']
# 讀取數字影像    
test_img = cv2.imread('8.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', test_img)
img = cv2.resize(test_img, (20, 20)).reshape((1, 400))
test_data = img.astype(np.float32)          # 將資料轉成foat32
# 最初化KNN或稱建立KNN物件，訓練數據、使用 k=5 測試KNN演算法
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(test_data, k=5)
print(f"識別的數字是 = {int(result[0,0])}")








