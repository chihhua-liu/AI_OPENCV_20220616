# ch21_11.py
import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = ["Microsoft JhengHei"] 

src = cv2.imread('jk.jpg',cv2.IMREAD_GRAYSCALE)
# 傅立葉變換
dft = cv2.dft(np.float32(src),flags=cv2.DFT_COMPLEX_OUTPUT) 
dftshift = np.fft.fftshift(dft)         # 0 頻率分量移至中心
# 低通濾波器
rows, cols = src.shape                  # 取得影像外形
row, col = rows // 2, cols // 2         # rows, cols的中心
mask = np.zeros((rows,cols,2),np.uint8)
mask[row-30:row+30,col-30:col+30] = 1   # 設定區塊為低頻率分量是1

fshift = dftshift * mask
ifshift = np.fft.ifftshift(fshift)      # 0 頻率分量移回左上角
src_tmp = cv2.idft(ifshift)             # 逆傅立葉
src_back = cv2.magnitude(src_tmp[:,:,0],src_tmp[:,:,1])              

plt.subplot(131)                        # 繪製左邊原圖
plt.imshow(src,cmap='gray')             # 灰階顯示
plt.title('原始影像')
plt.axis('off')                         # 不顯示座標軸
plt.subplot(132)                        # 繪製中間圖
plt.imshow(src_back,cmap='gray')        # 灰階顯示                 
plt.title('低通濾波灰階影像')
plt.axis('off')                         # 不顯示座標軸
plt.subplot(133)                        # 繪製右邊圖
plt.imshow(src_back)                    # 顯示                 
plt.title('低通濾波影像')
plt.axis('off')                         # 不顯示座標軸
plt.show()

