# import cv2 as cv2
# img1 = cv2.imread("jk.jpg")                 # 讀取影像
# print(f"成功讀取 : {type(img1)}")
# img2 = cv2.imread("none.jpg")               # 讀取影像
# print(f"讀取失敗 : {type(img2)}")
#------------------------------------
# import cv2 as cv2
# img = cv2.imread("C:/Users/mikal/PycharmProjects/AI_OPENCV_20220616/OpenCV_KingBook_program/code/ch1/jk.jpg")              # 讀取影像
# cv2.imshow("MyPicture", img)
# ret = cv2.waitKey(0)
# print(f"ret_value = {ret}")
# #cv2.destroyWindow("MyPicture")
# cv2.destroyAllWindows()
# #-----------------------------------------------------------
# import cv2
# img = cv2.imread("jk.jpg")              # 讀取影像
# cv2.imshow("MyPicture", img)            # 顯示影像
# ret_value = cv2.waitKey(0)              # 無限等待
# if ret_value == ord('Q') or ret_value == ord('q'):
#     cv2.destroyWindow("MyPicture")
# #-----------------------------------------------------------
# WINDOW_NORMAL: 可以調整大小, WINDOW_AUTOSIZE: 自動調整，不能調整大小 ㄝ
# import cv2
# cv2.namedWindow("MyPicture1")                       # 使用預設
# cv2.namedWindow("MyPicture2", cv2.WINDOW_NORMAL)    # 可以調整大小
# img1 = cv2.imread("jk.jpg")                         # 彩色讀取
# img2 = cv2.imread("jk.jpg", cv2.IMREAD_GRAYSCALE)   # 灰色讀取
# # img2 = cv2.imread("jk.jpg", 0)
# cv2.imshow("MyPicture1", img1)                      # 顯示影像img1
# cv2.imshow("MyPicture2", img2)                      # 顯示影像img2
# cv2.waitKey(3000)                                   # 等待3秒
# cv2.destroyWindow("MyPicture1")                     # 刪除MyPicture1
# cv2.waitKey(8000)                                   # 等待8秒
# cv2.destroyAllWindows()
#--------------------------------------------------------------
import cv2
cv2.namedWindow("MyPicture")            # 使用預設
img = cv2.imread("jk.jpg")              # 彩色讀取
cv2.imshow("MyPicture", img)            # 顯示影像img
ret = cv2.imwrite("out1_7_1.tiff", img) # 將檔案寫入out1_7_1.tiff
if ret:
    print("儲存檔案成功")
else:
    print("儲存檔案失敗")
ret = cv2.imwrite("out1_7_2.png", img)  # 將檔案寫入out1_7_2.png
if ret:
    print("儲存檔案成功")
else:
    print("儲存檔案失敗")
cv2.waitKey(3000)                       # 等待3秒
cv2.destroyAllWindows()                 # 刪除所有視窗