# ch14_8.py
import cv2

src = cv2.imread("pengiun.jpg")         # 讀取影像
G0 = src
G1 = cv2.pyrDown(G0)                    # 第 1 次向下採樣
L0 = src - cv2.pyrUp(G1)                # 拉普拉斯影像
dst = L0 + cv2.pyrUp(G1)                # 恢復結果影像

print(f"src.shape = \n{src.shape}")     # 列印原始影像大小
print(f"dst.shape = \n{dst.shape}")     # 列印恢復影像大小
cv2.imshow("Src",src)                   # 顯示原始影像
cv2.imshow("Dst",dst)                   # 顯示恢復影像

cv2.waitKey(0)
cv2.destroyAllWindows()


