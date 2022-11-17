# ch20_3.py
import cv2

src = cv2.imread("g5.jpg", cv2.IMREAD_COLOR)
temp1 = cv2.imread("face1.jpg", cv2.IMREAD_COLOR)
height, width = temp1.shape[:2]                     # 獲得模板影像的高與寬
# 使用cv2.TM_SQDIFF_NORMED執行模板匹配
result = cv2.matchTemplate(src, temp1, cv2.TM_SQDIFF_NORMED)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
upperleft = minLoc                                          # 左上角座標
lowerright = (minLoc[0] + width, minLoc[1] + height)        # 右下角座標
dst = cv2.rectangle(src,upperleft,lowerright,(0,255,0),3)   # 繪置最相似外框
cv2.imshow("Dst", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()


