# ch23_2.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft JhengHei"]

src = cv2.imread('hung.jpg')                # 讀取影像
mask = np.zeros(src.shape[:2],np.uint8)     # 建立遮罩, 大小和src相同
bgdModel = np.zeros((1,65),np.float64)      # 建立內部用暫時計算陣列
fgdModel = np.zeros((1,65),np.float64)      # 建立內部用暫時計算陣列
rect = (10,30,380,360)                      # 建立ROI區域
# 呼叫grabCut()進行分割 
cv2.grabCut(src,mask,rect,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)
maskpict = cv2.imread('hung_mask.jpg')      # 讀取影像
newmask = cv2.imread('hung_mask.jpg',cv2.IMREAD_GRAYSCALE)  # 灰階讀取
mask[newmask == 0] = 0                      # 白色內容則確定是前景
mask[newmask == 255] = 1                    # 黑色內容則確定是背景
cv2.grabCut(src,mask,None,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_MASK)
mask = np.where((mask==0)|(mask==2),0,1).astype('uint8')
dst = src * mask[:,:,np.newaxis]               # 計算輸出影像
src_rgb = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)   # 將BGR轉RGB
maskpict_rgb = cv2.cvtColor(maskpict,cv2.COLOR_BGR2RGB)
dst_rgb = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)   # 將BGR轉RGB
plt.subplot(131)
plt.title("原始影像")
plt.imshow(src_rgb)
plt.axis('off')
plt.subplot(132)
plt.title("遮罩影像")
plt.imshow(maskpict_rgb)
plt.axis('off')
plt.subplot(133)
plt.title("擷取影像")
plt.imshow(dst_rgb)
plt.axis('off')
plt.show()






