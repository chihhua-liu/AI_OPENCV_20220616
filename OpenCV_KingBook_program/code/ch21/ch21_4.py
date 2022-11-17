# ch21_4.py
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = ["Microsoft JhengHei"] # 正黑體

amplitude = [0,0,0,1,0,0,0]                      
N = len(amplitude)
x = np.arange(N)
width = 0.3
plt.bar(x,amplitude,width)                # 直條圖
plt.xlabel("頻率")                        # 頻率
plt.ylabel("振幅")                        # 振幅
plt.xticks(x,('1','2','3','4','5','6','7'))
plt.grid(axis="y")
plt.show()


