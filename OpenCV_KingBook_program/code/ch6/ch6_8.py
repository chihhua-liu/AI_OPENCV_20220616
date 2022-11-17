# ch6_8.py
import cv2

img = cv2.imread("street.png",cv2.IMREAD_UNCHANGED)     # PNG讀取
cv2.imshow("Before modify", img)        # 顯示修改前影像img
print(f"修改前img[10,50] = {img[10,50]}")
print(f"修改前img[50,99] = {img[50,99]}")
print("-"*70)
for z in range(0,200):                  # 一次一個修改alpha通道值
    for y in range(0,200):
        img[z,y,3] = 128                # 修改alpha通道值
print(f"修改後img[10,50] = {img[10,50]}")
print(f"修改後img[50,99] = {img[50,99]}")
cv2.imwrite("street128.png", img)       # 儲存含alpha通道的檔案

cv2.waitKey(0)                                      
cv2.destroyAllWindows()                 # 刪除所有視窗






 
