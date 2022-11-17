# demo23 : cv2.adaptiveThreshold 能夠幫助我們將一張圖片做自適應的二值化，
# cv2.adaptiveThreshold(image, 255, 自適應二值化算法, 閥值類型, 參考局部大小, 偏移量, dst=None) # dst is dstImage
# (1)自適應二值化算法:
# 1.cv2.ADAPTIVE_THRESH_MEAN_C，取的是「區域平均值 -C: C is 9」
# 2.cv2.ADAPTIVE_THRESH_GAUSSIAN_C，取的是「高斯平均值-C: C is 9」(高斯分佈)
# (2)閥值類型 : cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV
# 參考局部大小: 23 取單數
# 自適應二值化 搭配模糊降噪:cv2.medianBlur，能有更好的效果
# import cv2
# import matplotlib.pyplot as plt
# import time
# import numpy as np
#
# FILENAME = 'images/grayscale1.jpg'
#
# sourceImage = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
#
# THRESHOLD_VALUE = 128
# MAX_VALUE = 255
# th, ret1 = cv2.threshold(sourceImage, THRESHOLD_VALUE, MAX_VALUE, cv2.THRESH_BINARY)
# # ADAPTIVE_THRESH_MEAN_C is : threshold =取平均值-c
# ret2 = cv2.adaptiveThreshold(sourceImage, MAX_VALUE, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 9)
# # ADAPTIVE_THRESH_GAUSSIAN_C : threshold =取高斯權重-c
# ret3 = cv2.adaptiveThreshold(sourceImage, MAX_VALUE, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, 9)
# print(type(ret2), type(ret3))
# print(ret2.shape, ret3.shape, ret1.shape)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # demo23'
# import cv2
# from matplotlib import pyplot as plt
# import time
# import numpy as np
#
# FILENAME = 'images/grayscale1.jpg'
#
# sourceImage = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
#
# THRESHOLD_VALUE = 128
# MAX_VALUE = 255
# th, ret1 = cv2.threshold(sourceImage, THRESHOLD_VALUE, MAX_VALUE, cv2.THRESH_BINARY)
# ret2 = cv2.adaptiveThreshold(sourceImage, MAX_VALUE, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 9)
# ret3 = cv2.adaptiveThreshold(sourceImage, MAX_VALUE, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, 9)
# print(type(ret2), type(ret3))
# print(ret2.shape, ret3.shape, ret1.shape)
# titles = ['original',
#           'global thr.=%d' % THRESHOLD_VALUE,
#           'adaptive mean block=23, c=9',
#           'adaptive gaussian block=23, c=9']
# images = [sourceImage, ret1, ret2, ret3]
# for i in range(len(images)):
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

#======================================================================================
# # 模糊 & 平滑處理 (blur,medianBlur) :1、均值平滑blur() 2、中值平滑medianBlur()  3、像素值对比
# # ex : blur() :比如选择一个ksize=5×5的窗口，新图像的(x，y)点的像素值用numpy表示为 np.sum(i[x-2:x+3,y-2:y+3])/(5*5)
# #      dst=cv2.blur(src, ksize[, dst[, anchor[, borderType]]])
# import matplotlib.pyplot as plt
# import cv2
# print('VX公众号: 桔子code / juzicode.com')
# print('cv2.__version__:',cv2.__version__)
# plt.rc('font',family='Youyuan',size='9')
# FILENAME = 'images/grayscale1.jpg'
#
# img = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
# img_ret1 = cv2.blur(img,(3,3))
# img_ret2 = cv2.blur(img,(5,5))
# img_ret3 = cv2.blur(img,(11,11))
#
# #显示图像
# fig,ax = plt.subplots(2,2)
# ax[0,0].set_title('原图')
# ax[0,0].imshow(img,'gray')#matplotlib显示图像为rgb格式
# ax[0,1].set_title('blur ksize=3')
# ax[0,1].imshow(img_ret1,'gray')
# ax[1,0].set_title('blur ksize=5')
# ax[1,0].imshow(img_ret2,'gray')
# ax[1,1].set_title('blur ksize=11')
# ax[1,1].imshow(img_ret3,'gray')
# ax[0,0].axis('off');ax[0,1].axis('off');ax[1,0].axis('off');ax[1,1].axis('off')#关闭坐标轴显示
# plt.show()
# =====================================================================================
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
#
# FILENAME = 'images/lena_color.jpg'
# img = cv2.imread(FILENAME,1)
# cv2.imshow("original image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# img_ret1 = cv2.blur(img, (5, 5))
# cv2.imshow("img_ret1", img_ret1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# x, y = 10, 10
# b, g, r = cv2.split(img[x - 2:x + 3, y - 2:y + 3])  # 分离出原图在x，y点的bgr通道
# print('img b:\n', b)
# print('img g:\n', g)
# print('img r:\n', r)
# print('img average(b,g,r):', np.sum(b) / 25, np.sum(g) / 25, np.sum(r) / 25)  # 分别计算每个通道的平均值
# print('img_ret1[x,y]', img_ret1[x, y])
# =============================================================================
# # cv2.medianBlur(img,3) 使用原图像滑动窗口中所有像素值排序后的中值作为新图像的像素值,3*3
# import matplotlib.pyplot as plt
# import cv2
#
# plt.rc('font',family='Youyuan',size='9')
# FILENAME = 'images/lena_color.jpg'
# img = cv2.imread(FILENAME,1)
# print(img.shape)
# img_ret1 = cv2.medianBlur(img,3) # 3*3
# img_ret2 = cv2.medianBlur(img,5) # 5*5
# img_ret3 = cv2.medianBlur(img,11) # 11*11
#
# #显示图像
# fig,ax = plt.subplots(2,2)
# ax[0,0].set_title('原图')
# ax[0,0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) #matplotlib显示图像为rgb格式
# ax[0,1].set_title('medianBlur ksize=3')
# ax[0,1].imshow(cv2.cvtColor(img_ret1,cv2.COLOR_BGR2RGB))
# ax[1,0].set_title('medianBlur ksize=5')
# ax[1,0].imshow(cv2.cvtColor(img_ret2,cv2.COLOR_BGR2RGB))
# ax[1,1].set_title('medianBlur ksize=11')
# ax[1,1].imshow(cv2.cvtColor(img_ret3,cv2.COLOR_BGR2RGB))
# ax[0,0].axis('off');ax[0,1].axis('off');ax[1,0].axis('off');ax[1,1].axis('off')#关闭坐标轴显示
# plt.show()

# 像素值對比 ========================================
# # uint8（CV_8U）类型的数据
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
#
# plt.rc('font', family='Youyuan', size='10')
# FILENAME ="images/lena_color.jpg"
# img = cv2.imread(FILENAME,1)
# print(img.shape)
# img_ret1 = cv2.blur(img, (15, 15))
# img_ret2 = cv2.medianBlur(img, 15)
#
# X = np.arange(img.shape[1])
# #******************************************
# Y = img[:, :, 0][10, :]  # 仅提取b通道第10行
# print(Y.shape)              # img_ret1[:, :, 0][10, :]
# Y1 = img_ret1[:, :, 0][10, :]  # 提取均值平滑后图像b通道的第10行
# Y2 = img_ret2[:, :, 0][10, :]  # 提取中值平滑后图像b通道的第10行
# #******************************************
# plt.plot(X, Y, '-g', label='raw')  # 绘图
# plt.plot(X, Y1, '-r', label='blur')
# plt.plot(X, Y2, '-b', label='medianBlur')
# plt.legend(title='img type(juzicode.com)', fontsize='xx-large', loc='upper center')
# plt.show()
#==========================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
#
# FILENAME = "images/grayscale1.jpg"
# sourceImage = cv2.imread(FILENAME,cv2.IMREAD_GRAYSCALE)
# print(sourceImage.shape)
# THRESHOLD_VALUE = 128
# MAX_VALUE = 255
# sourceImage  = cv2.medianBlur(sourceImage,11)
#
# th, ret1 = cv2.threshold(sourceImage , THRESHOLD_VALUE, MAX_VALUE, cv2.THRESH_BINARY)
# ret2 = cv2.adaptiveThreshold(sourceImage , MAX_VALUE, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 9)
# ret3 = cv2.adaptiveThreshold(sourceImage , MAX_VALUE, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, 9)
# print(type(ret2), type(ret3))
# print(ret2.shape, ret3.shape, ret1.shape)
# titles = ['original',
#           'global thr.=%d' % THRESHOLD_VALUE,
#           'adaptive mean block=23, c=9',
#           'adaptive gaussian block=23, c=9']
# images = [sourceImage , ret1, ret2, ret3]
# for i in range(len(images)):
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()