# ch23_1.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft JhengHei"]

src = cv2.imread('hung.jpg')                # 讀取影像
mask = np.zeros(src.shape[:2],np.uint8)     # 建立遮罩, 大小和src相同
bgdModel = np.zeros((1,65),np.float64)      # 建立內部用暫時計算陣列
fgdModel = np.zeros((1,65),np.float64)      # 建立內部用暫時計算陣列
rect = (10,30,380,360)                      # 建立ROI區域
# 呼叫grabCut()進行分割, 迭代 3 次, 回傳mask1
# 其實mask1 = mask, 因為mask也會同步更新 
mask1, bgd, fgd = cv2.grabCut(src,mask,rect,bgdModel,fgdModel,3,
                             cv2.GC_INIT_WITH_RECT)
# 將 0, 2設為0 --- 1, 3設為1
mask2 = np.where((mask1==0)|(mask1==2),0,1).astype('uint8')
dst = src * mask2[:,:,np.newaxis]               # 計算輸出影像
src_rgb = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)   # 將BGR轉RGB
dst_rgb = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)   # 將BGR轉RGB
plt.subplot(121)
plt.title("原始影像")
plt.imshow(src_rgb)
plt.axis('off')
plt.subplot(122)
plt.title("擷取影像")
plt.imshow(dst_rgb)
plt.axis('off')
plt.show()






