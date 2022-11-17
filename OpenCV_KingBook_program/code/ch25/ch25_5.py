# ch25_5.py
import numpy as np
import matplotlib.pyplot as plt

num = 30                                            # 數據數量
np.random.seed(5)
trains = np.random.randint(0, 100, size = (num, 2))
np.random.seed(1)
# 建立分類, 未來 0 代表 red,  1 代表 blue
labels = np.random.randint(0,2,(num,1))
# 列出紅色方塊訓練數據
red = trains[labels.ravel() == 0]
plt.scatter(red[:, 0], red[:, 1], 50, 'r', 's')     # 50是繪圖點大小
# 列出藍色三角形訓練數據
blue = trains[labels.ravel() == 1]
plt.scatter(blue[:, 0], blue[:, 1], 50, 'b', '^')   # 50是繪圖點大小

plt.show()










