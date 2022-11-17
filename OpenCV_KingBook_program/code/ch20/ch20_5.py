# ch20_5.py
import cv2

src = cv2.imread("mutishapes.jpg", cv2.IMREAD_COLOR)
cv2.imshow("Src", src)                              # 顯示原始影像
temp1 = cv2.imread("heart.jpg", cv2.IMREAD_COLOR)
cv2.imshow("Temp1", temp1)                          # 顯示模板影像
height, width = temp1.shape[:2]                     # 獲得模板影像的高與寬
# 使用cv2.TM_CCOEFF_NORMED執行模板匹配
result = cv2.matchTemplate(src, temp1, cv2.TM_CCOEFF_NORMED)
for row in range(len(result)):                      # 找尋row
    for col in range(len(result[row])):             # 找尋column         
        if result[row][col] > 0.95:                 # 值大於0.95就算找到了
            dst = cv2.rectangle(src,(col,row),(col+width,row+height),(0,255,0),3)
cv2.imshow("Dst",dst)                     

cv2.waitKey(0)
cv2.destroyAllWindows()






