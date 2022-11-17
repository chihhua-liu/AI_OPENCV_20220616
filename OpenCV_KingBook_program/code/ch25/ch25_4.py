# ch25_4.py
import numpy as np

np.random.seed(1)
trains = np.random.randint(0, 10, size = (5, 2))
print(f"列出二維陣列 \n{trains}")
np.random.seed(5)
# 建立分類, 未來 0 代表 red,  1 代表 blue
labels = np.random.randint(0,2,(5,1))
print(f"列出顏色分類陣列 \n{labels}")
# 列出 0 代表的紅色
red = trains[labels.ravel() == 0]
print(f"輸出紅色的二維陣列 \n{red}")
print(f"配對取出 \n{red[:,0], red[:,1]}")
# 列出 1 代表的藍色
blue = trains[labels.ravel() == 1]
print(f"輸出藍色的二維陣列 \n{blue}")
print(f"配對取出 \n{blue[:,0], blue[:,1]}")








