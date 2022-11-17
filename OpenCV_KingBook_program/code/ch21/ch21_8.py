# ch21_8.py
import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = ["Microsoft JhengHei"] 

src = cv2.imread('snow.jpg',cv2.IMREAD_GRAYSCALE)
# 傅立葉變換
f = np.fft.fft2(src)                    # 轉成頻率域
fshift = np.fft.fftshift(f)             # 0 頻率分量移至中心
# 高通濾波器
rows, cols = src.shape                  # 取得影像外形
row, col = rows // 2, cols // 2         # rows, cols的中心
fshift[row-30:row+30,col-30:col+30] = 0 # 設定區塊為低頻率分量是0
# 逆傅立葉變換
ifshift = np.fft.ifftshift(fshift)      # 0 頻率分量移回左上角
src_tmp = np.fft.ifft2(ifshift)         # 逆傅立葉
src_back = np.abs(src_tmp)              # 取絕對值

plt.subplot(131)                        # 繪製左邊原圖
plt.imshow(src,cmap='gray')             # 灰階顯示
plt.title('原始影像')
plt.axis('off')                         # 不顯示座標軸
plt.subplot(132)                        # 繪製中間圖
plt.imshow(src_back,cmap='gray')        # 灰階顯示                 
plt.title('高通濾波灰階影像')
plt.axis('off')                         # 不顯示座標軸
plt.subplot(133)                        # 繪製右邊圖
plt.title('高通濾波影像')
plt.imshow(src_back)                    # 顯示影像
plt.axis('off')                         # 不顯示座標軸
plt.show()

