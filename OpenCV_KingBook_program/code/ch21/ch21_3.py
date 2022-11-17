# ch21_3.py
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = ["Microsoft JhengHei"] # 正黑體
plt.rcParams["axes.unicode_minus"] = False           # 可以顯示負數

start = 0
end = 1
x = np.linspace(start, end, 500)      # x 軸區間
y = np.sin(2*np.pi*4*x)               # 建立正弦曲線
plt.plot(x, y)
plt.xlabel("時間(秒)")                # 時間
plt.ylabel("振幅")                    # 振幅
plt.title("正弦曲線",fontsize=16)     # 標題
plt.show()




