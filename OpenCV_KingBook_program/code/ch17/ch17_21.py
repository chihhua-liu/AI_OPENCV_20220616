# ch17_21.py
import cv2

src = cv2.imread('forest.png')  
cv2.imshow("src",src)
mean, std = cv2.meanStdDev(src)          # 計算標準差
print(f"均值   = \n{mean}")
print(f"標準差 = \n{std}")

cv2.waitKey(0)
cv2.destroyAllWindows()



