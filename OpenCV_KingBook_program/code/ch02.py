# import cv2
# img = cv2.imread("jk.jpg", cv2.IMREAD_GRAYSCALE)   # 灰階讀取
# print("列印灰階影像的屬性")
# print(f"gray shape = {img.shape}")
# print(f"gray size  = {img.size}")    # = rows*columns* channels
# print(f"gray dtype = {img.dtype}")
#
# img = cv2.imread("jk.jpg")              # 彩色讀取
# print("列印彩色影像的屬性")
# print(f"color shape = {img.shape}")
# print(f"color size  = {img.size}")
# print(f"color dtype = {img.dtype}")
#---------------------------------------------------
# read pixels value
# import cv2
# pt_y = 169
# pt_x = 118
# img = cv2.imread("jk.jpg", cv2.IMREAD_GRAYSCALE)    # 灰階讀取
# px = img[pt_y, pt_x]                                # 讀px點
# print(type(px))
# print(f"gray BGR = {px}")
#
# img1 = cv2.imread("jk.jpg")      # 彩色讀取
# px_color = img1[pt_y, pt_x]            # 讀px點
# print(type(px_color))
# print(f"color BGR = {px_color}")
#--------------------------------------------------
# import cv2
# pt_y = 169
# pt_x = 118
# img = cv2.imread("jk.jpg")      # 彩色讀取
# blue = img[pt_y, pt_x, 0]       # 讀 B 通道值
# green = img[pt_y, pt_x, 1]      # 讀 G 通道值
# red = img[pt_y, pt_x, 2]        # 讀 R 通道值
# print(f"BGR = {blue}, {green}, {red}")
#---------------------------------------------------
# import cv2
# pt_y = 169
# pt_x = 118
# img = cv2.imread("jk.jpg")      # 彩色讀取
# px = img[pt_y, pt_x]            # 讀取 px 點
# print(f"更改前BGR = {px}")
# px = [255, 255, 255]            # 修改 px 點
# print(f"更改後BGR = {px}")
# -------------------------------------------------------
# import cv2
# # OpenCV 使用(y,x) 回傳像素資料
# img = cv2.imread("jk.jpg")      # 彩色讀取
# cv2.imshow("Before the change", img)
# for y in range(img.shape[0]-50, img.shape[0]):
#     for x in range(img.shape[1]-50, img.shape[1]):
#         img[y, x] = [255, 255, 255]
# cv2.imshow("After the change", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()