# ch6_8_1.py
import cv2

img = cv2.imread("street.png",cv2.IMREAD_UNCHANGED)     # PNG讀取
cv2.imshow("Before modify", img)        # 顯示修改前影像img
print(f"修改前img[10,50] = {img[10,50]}")
print(f"修改前img[50,99] = {img[50,99]}")
print("-"*70)
img[0:200,0:200,3] = 128
print(f"修改後img[10,50] = {img[10,50]}")
print(f"修改後img[50,99] = {img[50,99]}")
cv2.imwrite("street128_1.png", img)     # 儲存含alpha通道的檔案

cv2.waitKey(0)                                      
cv2.destroyAllWindows()                 # 刪除所有視窗






 
