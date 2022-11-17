# ch6_12.py
import cv2

img = cv2.imread("jk.jpg")              # 彩色讀取
cv2.imshow("Before modify", img)        # 顯示修改前影像img
print(f"修改前img[115,110,1] = {img.item(115,110,1)}")
print(f"修改前img[125,110,1] = {img.item(125,110,1)}")
print(f"修改前img[135,110,1] = {img.item(135,110,1)}")
# 白色長條
for z in range(115,145):                # 修改影像:一次一個通道值
    for y in range(110,210):
        for x in range(0,3):            # 一次一個通道值
            img.itemset((z,y,x),255)    # 白色取代    
cv2.imshow("After modify", img)         # 顯示修改後影像img
print(f"修改後img[115,110,1] = {img.item(115,110,1)}")
print(f"修改後img[125,110,1] = {img.item(125,110,1)}")
print(f"修改後img[135,110,1] = {img.item(135,110,1)}")
cv2.waitKey(0)                                      
cv2.destroyAllWindows()                 # 刪除所有視窗






 
