# ch14_1.py
import cv2

src = cv2.imread("macau.jpg")           # 讀取影像
dst1 = cv2.pyrDown(src)                 # 第 1 次向下採樣
dst2 = cv2.pyrDown(dst1)                # 第 2 次向下採樣
dst3 = cv2.pyrDown(dst2)                # 第 3 次向下採樣
print(f"src.shape = {src.shape}")
print(f"dst1.shape = {dst1.shape}")
print(f"dst2.shape = {dst2.shape}")
print(f"dst3.shape = {dst3.shape}")

cv2.imshow("src",src)
cv2.imshow("dst1",dst1)
cv2.imshow("dst2",dst2)
cv2.imshow("dst3",dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()

