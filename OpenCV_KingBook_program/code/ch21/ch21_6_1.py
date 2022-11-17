# ch21_6_1.py
import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = ["Microsoft JhengHei"] 

src = cv2.imread('shape1.jpg',cv2.IMREAD_GRAYSCALE)
f = np.fft.fft2(src)                    # 轉成頻率域
fshift = np.fft.fftshift(f)             # 0 頻率分量移至中心     
spectrum = 20*np.log(np.abs(fshift))    # 轉成頻譜
plt.subplot(121)                        # 繪製左邊原圖
plt.imshow(src,cmap='gray')             # 灰階顯示
plt.title('原始影像shape1.jpg')
plt.axis('off')                         # 不顯示座標軸
plt.subplot(122)                        # 繪製右邊頻譜圖
plt.imshow(spectrum,cmap='gray')        # 灰階顯示
plt.title('頻譜圖')
plt.axis('off')                         # 不顯示座標軸
plt.show()

