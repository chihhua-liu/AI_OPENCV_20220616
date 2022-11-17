# ch6_2.py
import cv2

img = cv2.imread("jk.jpg", cv2.IMREAD_GRAYSCALE)    # 灰色讀取
cv2.imshow("Before modify", img)                    # 顯示修改前影像img
for y in range(120,140):                            # 修改影像
    for x in range(110,210):
        img[y,x] = 255
cv2.imshow("After modify", img)                     # 顯示修改後影像img

cv2.waitKey(0)                                      
cv2.destroyAllWindows()                             # 刪除所有視窗






 
