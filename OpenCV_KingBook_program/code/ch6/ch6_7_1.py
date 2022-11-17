# ch6_7_1.py
import cv2

img = cv2.imread("jk.jpg")              # 彩色讀取
cv2.imshow("Before modify", img)        # 顯示修改前影像img
print(f"修改前img[115,110] = {img[115,110]}")
print(f"修改前img[125,110] = {img[125,110]}")
print(f"修改前img[135,110] = {img[135,110]}")
# 紫色長條
img[115:125,110:210] = [255, 0, 255]        
# 白色長條
for z in range(125,135):                # 修改影像:一次一個通道值
    for y in range(110,210):
        for x in range(0,3):            # 一次一個通道值
            img[z,y,x] = 255            # 白色取代
# 黃色長條
for y in range(135,145):                # 修改影像
    for x in range(110,210):
        img[y,x] = [0,255,255]          # 黃色取代        
cv2.imshow("After modify", img)         # 顯示修改後影像img
print(f"修改後img[115,110] = {img[115,110]}")
print(f"修改後img[125,110] = {img[125,110]}")
print(f"修改後img[135,110] = {img[135,110]}")
cv2.waitKey(0)                                      
cv2.destroyAllWindows()                 # 刪除所有視窗






 
