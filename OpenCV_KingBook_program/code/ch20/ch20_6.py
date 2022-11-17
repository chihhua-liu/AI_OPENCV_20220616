# ch20_6.py
import cv2

src = cv2.imread("baidu.jpg", cv2.IMREAD_COLOR)
temp1 = cv2.imread("mountain_mark.jpg", cv2.IMREAD_COLOR)
h, w = temp1.shape[:2]                              # 獲得模板影像的高與寬
# 使用cv2.TM_CCOEFF_NORMED執行模板匹配
result = cv2.matchTemplate(src, temp1, cv2.TM_CCOEFF_NORMED)
for row in range(len(result)):                      # 找尋row
    for col in range(len(result[row])):             # 找尋column         
        if result[row][col] > 0.95:                 # 值大於0.95就算找到了
            dst = cv2.rectangle(src,(col,row),(col+w,row+h),(0,0,255),3)
cv2.imshow("Dst",dst)                     

cv2.waitKey(0)
cv2.destroyAllWindows()






