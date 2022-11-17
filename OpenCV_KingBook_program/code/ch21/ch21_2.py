# ch21_2.py
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = ["Microsoft JhengHei"] # 正黑體

copies = [1,2,4,3]                      # 份數
N = len(copies)
x = np.arange(N)
width = 0.35
plt.bar(x,copies,width)                 # 直條圖
plt.xlabel("頻率")                      # 頻率
plt.ylabel("份數")                      # 份數
plt.xticks(x,('1','2','3','4'))
plt.grid(axis="y")
plt.show()


