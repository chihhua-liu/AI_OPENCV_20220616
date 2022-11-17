# ch21_5.py
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = ["Microsoft JhengHei"] # 正黑體
plt.rcParams["axes.unicode_minus"] = False           # 可以顯示負數

start = 0;                                           # 起始時間
end = 5;                                             # 結束時間
# 兩個正弦波的訊號頻率
freq1 = 5;                                           # 頻率是 5 Hz
freq2 = 8;                                           # 頻率是 8 Hz
# 建立時間軸的Numpy陣列, 用500個點
time = np.linspace(start, end, 500);
# 建立2個正弦波
amplitude1 = np.sin(2*np.pi*freq1*time)
amplitude2 = np.sin(2*np.pi*freq2*time)
# 建立子圖
figure, axis = plt.subplots(3,1)
plt.subplots_adjust(hspace=1)
# 時間域的 sin 波 1
axis[0].set_title('頻率是 5 Hz的 sin 波')
axis[0].plot(time, amplitude1)
axis[0].set_xlabel('時間')
axis[0].set_ylabel('振幅')
# 時間域的 sin 波 2
axis[1].set_title('頻率是 8 Hz的 sin 波')
axis[1].plot(time, amplitude2)
axis[1].set_xlabel('時間')
axis[1].set_ylabel('振幅')
# 加總sin波
amplitude = amplitude1 + amplitude2
axis[2].set_title('2個不同頻率正弦波的結果')
axis[2].plot(time, amplitude)
axis[2].set_xlabel('時間')
axis[2].set_ylabel('振幅')
plt.show()


