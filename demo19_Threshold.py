#######################################################################
# 1. pixels by pixels 非常慢 :6.5867869 sec
# image > threshold : value = 255，image < threshold : value = 0  ##
#######################################################################
# import cv2
# import matplotlib.pyplot as plt
# import time
# FILENAME = 'images/grayscale1.jpg'
#
# sourceImage = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
# # cv2.imshow("original image", sourceImage)
# # cv2.waitKey(0)
#
# THRESHOLD_VALUE = 50
# maxValue = 255
#
# def thresholdUsingLoop(image, threshold, max):
#     dst = image.copy()
#     print(image.shape[:2])
#     height, width = image.shape[:2]   # 先Height, 再width
#     for i in range(height):
#         for j in range(width):
#             if sourceImage[i, j] > threshold:
#                 dst[i, j] = maxValue
#             else:
#                 dst[i, j] = 0
#     return dst
#
# plt.imshow(sourceImage, cmap='gray', vmin=0, vmax=255)
# startTime = time.time()
# binaryImage = thresholdUsingLoop(sourceImage, THRESHOLD_VALUE, maxValue)
# endTime = time.time()
# print(f"it took {endTime - startTime} to do threshold")
# plt.show()
# plt.subplot(121)
# plt.title("original image")
# plt.imshow(sourceImage, cmap='gray', vmin=0, vmax=255)
# plt.subplot(122)
# plt.title("binary image")
# plt.imshow(binaryImage, cmap='gray', vmin=0, vmax=255)

# plt.show()
# ============================================================
##########################################################################
# 2. 用 Image matrix 作法快速 : 0.019829 sec 值得學習
# image > threshold : value =image.value，image < threshold : value = 0 ##
##########################################################################
# import cv2
# from matplotlib import pyplot
# import time
# import numpy as np
#
# FILENAME = 'images/grayscale1.jpg'
#
# sourceImage = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
# # cv2.imshow("original image", sourceImage)
# # cv2.waitKey(0)
#
# THRESHOLD_VALUE = 100
# MAX_VALUE = 255
#
#
# def thresholdUsingVector(image, threshold, v_max):  # 15.91
#     dst = np.zeros_like(image)     # np.zeros_like: 跟image 的長寬一樣，值全部=0
#     print(type(image), type(dst))
#     print(image.shape, dst.shape)
#     #*****************************************************************
#     maxPixels = image > threshold  # image 的 Value > threshold 的所有點
#     zeroPixels = image < threshold # image 的 Value < threshold 的所有點
#     #*****************************************************************
#     dst[maxPixels] = image[maxPixels]
#     dst[zeroPixels] = 0
#     return dst
#
# pyplot.imshow(sourceImage,cmap='gray',vmin=0,vmax=255)
# startTime = time.time()   # time.time() 可以傳回從 1970/1/1 00:00:00 算起至今的秒數
# binaryImage = thresholdUsingVector(sourceImage, THRESHOLD_VALUE, MAX_VALUE)
# endTime = time.time()
# print(f"it took {endTime - startTime} to do threshold")
# pyplot.show()
# pyplot.subplot(121)
# pyplot.title("original image")
# pyplot.imshow(sourceImage, cmap='gray', vmin=0, vmax=255)
# pyplot.subplot(122)
# pyplot.title("binary image")
# pyplot.imshow(binaryImage, cmap='gray', vmin=0, vmax=255)
# pyplot.show()

##########################################################################
# 3. 用 cv2.threshold 各種選擇參數大全 :  最快 :0.00584316
# cv2.threshold(img, 閥值, 最大灰度值, 使用的二值化方法)
##########################################################################
# import cv2
# from matplotlib import pyplot
# import time
# FILENAME = 'images/grayscale1.jpg'
#
# img = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
#
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 先將圖片轉為灰階
# # 將小於閾值的灰度值設為0，其他值設為最大灰度值。>127 =255, <127 =0
# start_time = time.time()
# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)  # ret is threshold
# end_time = time.time()
# print(f'cv2.threshold took time is{end_time-start_time}')
# print('ret=',ret)
# # 將大於閾值的灰度值設為0，其他值設為最大灰度值。>127 =0, <127 =255
# ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# print('ret=',ret)
# # 將大於閾值的灰度值設為閾值，小於閾值的值保持不變。 >127 =127
# ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# print('ret=',ret)
# # 將小於閾值的灰度值設為0，大於閾值的值保持不變。 <127 =0
# ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# print('ret=',ret)
# # 將大於閾值的灰度值設為0，小於閾值的值保持不變。 >127 =0
# ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
# print('ret=',ret)
#
# pyplot.subplot(231)
# pyplot.title("original image")
# pyplot.imshow(img, cmap='gray', vmin=0, vmax=255)
#
# pyplot.subplot(232)
# pyplot.title("THRESH_BINARY")
# pyplot.imshow(thresh1, cmap='gray', vmin=0, vmax=255)
#
# pyplot.subplot(233)
# pyplot.title("THRESH_BINARY_INV")
# pyplot.imshow(thresh2, cmap='gray', vmin=0, vmax=255)
#
# pyplot.subplot(234)
# pyplot.title("THRESH_TRUNC")
# pyplot.imshow(thresh3, cmap='gray', vmin=0, vmax=255)
#
# pyplot.subplot(235)
# pyplot.title("THRESH_TOZERO")
# pyplot.imshow(thresh4, cmap='gray', vmin=0, vmax=255)
#
# pyplot.subplot(236)
# pyplot.title("THRESH_TOZERO_INV")
# pyplot.imshow(thresh5, cmap='gray', vmin=0, vmax=255)
# pyplot.show()
#===================================================================
# import cv2
# from matplotlib import pyplot
# import time
# import numpy as np
#
# FILENAME = 'images/grayscale1.jpg'
#
# sourceImage = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
# # cv2.imshow("original image", sourceImage)
# # cv2.waitKey(0)
#
# THRESHOLD_VALUE = 50
# MAX_VALUE = 255
#
# pyplot.imshow(sourceImage,cmap='gray',vmin=0,vmax=255)
# startTime = time.time()
# threshold_return, binaryImage = cv2.threshold(sourceImage, THRESHOLD_VALUE, MAX_VALUE, cv2.THRESH_BINARY)
# # binaryImage = thresholdUsingVector(sourceImage, THRESHOLD_VALUE, MAX_VALUE)
# endTime = time.time()
# print(f"it took {endTime - startTime} to do threshold")
# pyplot.show()
# pyplot.subplot(121)
# pyplot.title("original image")
# pyplot.imshow(sourceImage, cmap='gray', vmin=0, vmax=255)
# pyplot.subplot(122)
# pyplot.title("binary image")
# pyplot.imshow(binaryImage, cmap='gray', vmin=0, vmax=255)
# pyplot.show()
#===========================================================
# https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

FILENAME = 'images/grayscale1.jpg'

sourceImage = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)

THRESHOLD_VALUE = 128
MAX_VALUE = 255

THRESHOLDS = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]
resultImages = []
for i, t in enumerate(THRESHOLDS):
    print(f"[{i}],threshold={t}")
    _, resultImg = cv2.threshold(sourceImage, THRESHOLD_VALUE, MAX_VALUE, t)
    resultImages.append(resultImg)

print(len(resultImages))
plt.subplot(2, 3, 1)
plt.title('original image')
plt.imshow(sourceImage, cmap='gray', vmin=0, vmax=255)
plt.subplot(2, 3, 2)
plt.title('binary thr. image')
plt.imshow(resultImages[0], cmap='gray', vmin=0, vmax=255)
plt.subplot(2, 3, 3)
plt.title('binary thr. inverse image')
plt.imshow(resultImages[1], cmap='gray', vmin=0, vmax=255)
plt.subplot(2, 3, 4)
plt.title('binary thr. truncate')
plt.imshow(resultImages[2], cmap='gray', vmin=0, vmax=255)
plt.subplot(2, 3, 5)
plt.title('binary thr. zero')
plt.imshow(resultImages[3], cmap='gray', vmin=0, vmax=255)
plt.subplot(2, 3, 6)
plt.title('binary thr. zero inverse')
plt.imshow(resultImages[4], cmap='gray', vmin=0, vmax=255)

plt.show()
