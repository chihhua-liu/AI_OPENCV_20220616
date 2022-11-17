# ch6_4.py
import cv2
import numpy as np
 
# 建立藍色blue底的彩色影像陣列
blue_img = np.zeros((100,150,3),np.uint8)   
blue_img[:,:,0] = 255                   # 填滿藍色
print(f"blue image =\n{blue_img}")      # 顯示blue_img影像陣列
cv2.imshow("Blue Image",blue_img)       # 顯示藍色影像

# 建立綠色green底的彩色影像陣列
green_img = np.zeros((100,150,3),np.uint8)   
green_img[:,:,1] = 255                  # 填滿綠色
print(f"green image =\n{green_img}")    # 顯示green_img影像陣列
cv2.imshow("Green Image",green_img)     # 顯示綠色影像

# 建立紅色red底的彩色影像陣列
red_img = np.zeros((100,150,3),np.uint8)   
red_img[:,:,2] = 255                    # 填滿紅色
print(f"red image =\n{red_img}")        # 顯示red_img影像陣列
cv2.imshow("Red Image",red_img)         # 顯示紅色影像

cv2.waitKey(0)                                      
cv2.destroyAllWindows()                 # 刪除所有視窗

 

