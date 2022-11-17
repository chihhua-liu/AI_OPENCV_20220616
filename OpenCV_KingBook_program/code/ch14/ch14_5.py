# ch14_5.py
import cv2

src = cv2.imread("pengiun.jpg")         # 讀取影像
print(f"原始影像大小 = \n{src.shape}")
dst_down = cv2.pyrDown(src)             # 向下採樣
print(f"向下採樣大小 = \n{dst_down.shape}")
dst_up = cv2.pyrUp(dst_down)             # 向上採樣, 復原大小
print(f"向上採樣大小 = \n{dst_up.shape}")
dst = dst_up - src
print(f"結果影像大小 = \n{dst.shape}")

cv2.imshow("src",src)
cv2.imshow("dst1 - recovery",dst_up)
cv2.imshow("dst2 - dst",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
