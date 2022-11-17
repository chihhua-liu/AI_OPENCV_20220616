# ch25_13.py
import cv2
import numpy as np

img = cv2.imread('digits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 將digits拆成 5000 張, 20 x 20 的數字影像
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
# 將 cells 轉成 50 x 100 x 20 x 20 的陣列
x = np.array(cells)
# 將數據轉為訓練數據 size=(2500,400)和測試數據 size=(2500,400)
train = x[:,:50].reshape(-1,400).astype(np.float32) 
test = x[:,50:100].reshape(-1,400).astype(np.float32) 
# 建立訓練數據和測試數據的分類 labels
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()
# 最初化KNN或稱建立KNN物件，訓練數據、使用 k=5 測試KNN演算法
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(test, k=5)
# 統計辨識結果
matches = result==test_labels               # 執行匹配
correct = np.count_nonzero(matches)         # 正確次數
accuracy = correct * 100.0 / result.size    # 精確度
print(f"測試數據辨識成功率 = {accuracy}")
np.savez('knn_digit.npz',train=train, train_labels=train_labels)








