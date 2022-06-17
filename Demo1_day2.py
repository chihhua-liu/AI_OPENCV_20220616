# import os
# import sys
# import cv2

# Sample ----------------------------------------------
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#     print(os.getcwd())
#     print(sys.executable)
#     print(cv2.__version__)
#-------------------------------------------------------
# demo1 : 回傳值 = cv2.imread(檔名路徑, 標記):
# 標記 cv2.IMREAD_GRAYSCALE = 0
# 標記 cv2.IMREAD_COLOR = 1
# 標記 cv2.IMREAD_UNCHANGED = 2
#(1)cv2.IMREAD_COLOR :
# #此為預設值，這種格式會讀取 RGB 三個 channels 的彩色圖片，而忽略透明度的 channel。
# (2)cv2.IMREAD_GRAYSCALE  :以灰階的格式來讀取圖片。
# (3) cv2.IMREAD_UNCHANGED : 讀取圖片中所有的 channels，包含透明度的 channel。

# import cv2
# # make a directory images
# # copy number_zero ==> images
# IMAGE_PATH1 = 'images/number_zero.jpg'
# image1 = cv2.imread(IMAGE_PATH1, -1)
# print(image1)
# print(image1.shape) #

# demo2 : plt.imshow(image1)-------------------------------
# import cv2
# import matplotlib
# import matplotlib.pyplot as plt
# # make a directory images
# # copy number_zero ==> images
# IMAGE_PATH1 = 'images/number_zero.jpg'
# image1 = cv2.imread(IMAGE_PATH1, 0)
# print(image1)
# print(type(image1))
# print(image1.dtype)    # <class 'numpy.ndarray'>
# print(image1[0][0], type(image1[0][0]))  #<class 'numpy.uint8'>
# print(image1.shape)  # (13, 11) gray_scale image
# image1[0][0]=255
# image1[0][1]=255
# image1[1][0]=255
# image1[1][1]=255
# print(image1)
# matplotlib.rcParams['image.cmap'] = 'gray'  # 沒有這一行 ，Show color image
# plt.imshow(image1)
# plt.colorbar()
# plt.show()

# demo3 -----------------------------------------------------------
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib
#
# # make a directory images
# # copy number_zero ==> images
# IMAGE_PATH1 = 'images/number_zero.jpg'
# image1 = cv2.imread(IMAGE_PATH1, 0)
# print(image1)
# print(type(image1)) # <class 'numpy.ndarray'> 影像的型態
# print(image1.dtype) # uint8  影像pixels 的型態
# print(image1[0][0], type(image1[0][0])) # <class 'numpy.uint8'>
# print(image1.shape)  # 維度
# image1[0][0] = 255
# image1[0][1] = 255
# image1[1][0] = 255
# image1[1][1] = 255
# print(image1)
# cv2.imshow("this is cv2.imshow", image1)  # 不show
# plt.imshow(image1)  # show 2
# plt.show()
#
# image1[0:6, 0:4] = 128
# plt.imshow(image1)
# plt.show()  #show 3
#
# matplotlib.rcParams['figure.figsize'] = (6.0, 6.0) # #圖片畫素
# matplotlib.rcParams['image.cmap'] = 'gray'
# plt.imshow(image1)  # show 4
# plt.colorbar()
# plt.show()

# demo04 : OpenCV 中 color image is BGR 需轉換成RGB----------------------------------------
# import cv2
# import matplotlib
# from matplotlib import pyplot as plt
#
# matplotlib.rcParams['image.cmap'] = 'gray'
# IMAGE1 = 'images/bg1.jpg'
# image1 = cv2.imread(IMAGE1)
# print(f"image dim={image1.shape}")
# plt.imshow(image1)
# plt.show()

#---demo05 由BGR 轉換 RGB 兩個方法  -------------
# (1) image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# (2)  image3 = image1[:, :, ::-1]
# import cv2
# import matplotlib
# from matplotlib import pyplot as plt
#
# matplotlib.rcParams['image.cmap'] = 'gray'
# IMAGE1 = 'images/bg1.jpg'
# image1 = cv2.imread(IMAGE1)
# print(f"image dim={image1.shape}")
# plt.title("default, but wrong rgb channel")
# plt.imshow(image1)
# plt.show()
#
# image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# plt.title("after convert")
# plt.imshow(image2)
# plt.show()
#
# image3 = image1[:, :, ::-1]
# plt.title("whole pictures, but BGR-->RGB")
# plt.imshow(image3)
# plt.show()

# demo06 -R, G, B image 個別show--------------------------------------
# import cv2
# import matplotlib
# from matplotlib import pyplot as plt
#
# matplotlib.rcParams['image.cmap'] = 'gray'
# IMAGE1 = 'images/bg1.jpg'
# image1 = cv2.imread(IMAGE1)
# print(f"image dim={image1.shape}")
# plt.title("default, but wrong rgb channel")
# plt.imshow(image1)
# plt.show()
#
# image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# plt.title("after convert")
# plt.imshow(image2)
# plt.show()
# image3 = image1[:, :, ::-1]
# plt.title("whole pictures, but BGR-->RGB")
# plt.imshow(image3)
# plt.show()
#
# #----R, G, B image 個別show-法1----
# plt.figure(figsize=[20,9])
# plt.subplot(131) # row 1, column 3 : 第1張 image
# plt.title('Blue channel')
# plt.imshow(image1[:,:,0])
# plt.subplot(132)    # row 1, column 3 : 第2張 image
# plt.title("Green channel")
# plt.imshow(image1[:,:,1])
# plt.subplot(133)    # row 1, column 3 : 第3張 image
# plt.title("Red channel")
# plt.imshow(image1[:,:,2])
# plt.show()
#
# #----R, G, B image 個別show--法2---
## b, g, r = cv2.split(image1)
# result = cv2.split(image1)
# plt.figure(figsize=[20, 9])
# plt.subplot(131)
# plt.title('Blue channel')
## plt.imshow(b)
#plt.imshow(result[0])
# plt.subplot(132)
# plt.title("Green channel")
## plt.imshow(g)
#plt.imshow(result[1])
# plt.subplot(133)
# plt.title("Red channel")
## plt.imshow(r)
# plt.imshow(result[2])
# plt.show()

#live template ------------------------
import cv2
import matplotlib.pyplot as plt

IMAGE1 = 'images/bg1.jpg'
orgImage =cv2.imread(IMAGE1, -1)
cv2.imshow("Original Image",orgImage) # cv2.imshow 自己會轉成正常的
plt.imshow(orgImage[: ,: ,::-1])
plt.title("Original Image")
plt.show()
# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

