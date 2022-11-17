# ch20_8.py
import cv2

def myMatch(image,tmp):
    ''' 執行匹配 '''
    h, w = tmp.shape[0:2]                               # 回傳height, width
    result = cv2.matchTemplate(src, tmp, cv2.TM_CCOEFF_NORMED)
    for row in range(len(result)):                      # 找尋row
        for col in range(len(result[row])):             # 找尋column         
            if result[row][col] > 0.95:                 # 值大於0.95就算找到了
                match.append([(col,row),(col+w,row+h)]) # 左上與右下點加入串列
    return 

src = cv2.imread("mutishapes1.jpg", cv2.IMREAD_COLOR)   # 讀取原始影像
temps = []
temp1 = cv2.imread("heart1.jpg", cv2.IMREAD_COLOR)      # 讀取匹配影像
temps.append(temp1)                                     # 加入匹配串列temps
temp2 = cv2.imread("star.jpg", cv2.IMREAD_COLOR)        # 讀取匹配影像
temps.append(temp2)                                     # 加入匹配串列temps
match = []                                              # 符合匹配的圖案
for t in temps:
    myMatch(src,t)                                      # 調用 myMatch
for img in match:
    dst = cv2.rectangle(src,(img[0]),(img[1]),(0,255,0),1)  # 繪外框 
cv2.imshow("Dst",dst)                     

cv2.waitKey(0)
cv2.destroyAllWindows()






