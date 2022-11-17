# ch21_9.py
import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = ["Microsoft JhengHei"] 

src = cv2.imread('jk.jpg',cv2.IMREAD_GRAYSCALE)
# 轉成頻率域
dft = cv2.dft(np.float32(src),flags=cv2.DFT_COMPLEX_OUTPUT) 
dftshift = np.fft.fftshift(dft)         # 0 頻率分量移至中心
# 計算映射到[0,255]的振幅
spectrum = 20*np.log(cv2.magnitude(dftshift[:,:,0],dftshift[:,:,1]))                     
plt.subplot(121)                        # 繪製左邊原圖
plt.imshow(src,cmap='gray')             # 灰階顯示
plt.title('原始影像')
plt.axis('off')                         # 不顯示座標軸
plt.subplot(122)                        # 繪製右邊頻譜圖
plt.imshow(spectrum,cmap='gray')        # 灰階顯示
plt.title('頻譜圖')
plt.axis('off')                         # 不顯示座標軸
plt.show()

