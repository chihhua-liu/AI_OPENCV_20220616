# ch16_17.py
import cv2
src = cv2.imread("heart1.jpg")
cv2.imshow("src",src)
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)     # 影像轉成灰階
# 二值化處理影像
ret, dst_binary = cv2.threshold(src_gray,127,255,cv2.THRESH_BINARY)
# 找尋影像內的輪廓
contours, hierarchy = cv2.findContours(dst_binary,
                      cv2.RETR_LIST,
                      cv2.CHAIN_APPROX_SIMPLE)
# 凸包
hull = cv2.convexHull(contours[0])                  # 獲得凸包頂點座標       
dst = cv2.polylines(src, [hull], True, (0,255,0),2) # 將凸包連線
# print(hull)   可以用這個指令了解凸包座標點
# 點在凸包線上
pointa = (231,85)                                   # 點在凸包線上
dist_a = cv2.pointPolygonTest(hull,pointa, False)   # 檢測距離 
font = cv2.FONT_HERSHEY_SIMPLEX
pos_a = (236,95)                                    # 文字輸出位置
dst = cv2.circle(src,pointa,3,[0,0,255],-1)         # 用圓標記點 A
cv2.putText(dst,'A',pos_a,font,1,(0,255,255),2)     # 輸出文字 A
print(f"dist_a = {dist_a}") 
# 點在凸包內
pointb = (150,100)                                  # 點在凸包線上
dist_b = cv2.pointPolygonTest(hull,pointb, False)   # 檢測距離 
font = cv2.FONT_HERSHEY_SIMPLEX
pos_b = (160,110)                                   # 文字輸出位置
dst = cv2.circle(src,pointb,3,[0,0,255],-1)         # 用圓標記點 B
cv2.putText(dst,'B',pos_b,font,1,(255,0,0),2)       # 輸出文字 B
print(f"dist_b = {dist_b}") 
# 點在凸包外
pointc = (80,85)                                    # 點在凸包線上
dist_c = cv2.pointPolygonTest(hull,pointc, False)   # 檢測距離 
font = cv2.FONT_HERSHEY_SIMPLEX
pos_c = (50,95)                                     # 文字輸出位置
dst = cv2.circle(src,pointc,3,[0,0,255],-1)         # 用圓標記點 C
cv2.putText(dst,'C',pos_c,font,1,(0,255,255),2)     # 輸出文字 C
print(f"dist_c = {dist_c}") 
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()




