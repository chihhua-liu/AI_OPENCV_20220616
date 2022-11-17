# ch20_7.py
import cv2
import math

start_x = 450                                       # 目前位置 x
start_y = 180                                       # 目前位置 y
src = cv2.imread("airport.jpg", cv2.IMREAD_COLOR)
temp1 = cv2.imread("airport_mark.jpg", cv2.IMREAD_COLOR)
dst = cv2.circle(src,(start_x,start_y),10,(255,0,0),-1)
h, w = temp1.shape[:2]                              # 獲得模板影像的高與寬
# 使用cv2.TM_CCOEFF_NORMED執行模板匹配
ul_x = []                                           # 最佳匹配左上角串列 x
ul_y = []                                           # 最佳匹配左上較串列 y
result = cv2.matchTemplate(src, temp1, cv2.TM_CCOEFF_NORMED)
for row in range(len(result)):                      # 找尋row
    for col in range(len(result[row])):             # 找尋column         
        if result[row][col] > 0.9:                  # 值大於0.9就算找到了
            dst = cv2.rectangle(src,(col,row),(col+w,row+h),(255,0,0),2)
            ul_x.append(col)                        # 加入最佳匹配串列 x
            ul_y.append(row)                        # 加入最佳匹配串列 y
# 計算目前位置到台北機場的距離
sub_x = start_x - ul_x[0]                           # 計算 x 座標差距
sub_y = start_y - ul_y[0]                           # 計算 y 座標差距
start_taipei = math.hypot(sub_x,sub_y)              # 計算距離
print(f"目前位置到台北機場的距離 = {start_taipei:8.2f}")
# 計算目前位置到桃園機場的距離
sub_x = start_x - ul_x[1]                           # 計算 x 座標差距
sub_y = start_y - ul_y[1]                           # 計算 y 座標差距
start_taoyuan = math.hypot(sub_x,sub_y)             # 計算距離
print(f"目前位置到桃園機場的距離 = {start_taoyuan:8.2f}")
# 計算最短距離
if start_taipei > start_taoyuan:                    # 距離比較
    cv2.line(src,(start_x,start_y),(ul_x[0],ul_y[0]),(255,0,0),2)
else:
    cv2.line(src,(start_x,start_y),(ul_x[1],ul_y[1]),(255,0,0),2)               
cv2.imshow("Dst",dst)                     

cv2.waitKey(0)
cv2.destroyAllWindows()






