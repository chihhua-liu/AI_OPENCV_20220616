# ch21_7.py
import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = ["Microsoft JhengHei"] 

src = cv2.imread('jk.jpg',cv2.IMREAD_GRAYSCALE)
# 傅立葉變換
f = np.fft.fft2(src)                    # 轉成頻率域
fshift = np.fft.fftshift(f)             # 0 頻率分量移至中心
# 逆傅立葉變換
ifshift = np.fft.ifftshift(fshift)      # 0 頻率頻率移回左上角
src_tmp = np.fft.ifft2(ifshift)         # 逆傅立葉
src_back = np.abs(src_tmp)              # 取絕對值

plt.subplot(121)                        # 繪製左邊原圖
plt.imshow(src,cmap='gray')             # 灰階顯示
plt.title('原始影像')
plt.axis('off')                         # 不顯示座標軸
plt.subplot(122)                        # 繪製右邊逆運算圖
plt.imshow(src_back,cmap='gray')        # 灰階顯示
plt.title('逆變換影像')
plt.axis('off')                         # 不顯示座標軸
plt.show()

