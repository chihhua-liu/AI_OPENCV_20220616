# demo24  #常用的 kernal (rect/elliptical/cross)
#
import cv2

import matplotlib.pyplot as plt

kernels = [cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS]

plt.figure(figsize=[15, 15])
for i, k in zip([1, 2, 3], kernels):
    a1 = cv2.getStructuringElement(k, (15, 15))
    plt.subplot(1, 3, i)
    plt.imshow(a1, 'gray')
    print(a1)
plt.show()
#============================
# Morphological Transformations
# 1.Erosion ===========================================
# import cv2 as cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# img = cv2.imread('images/j.png',0)
# print(img.shape)
# cv2.imshow('j_png',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(img,kernel,iterations = 1) # iterations = 1, 左右個縮1 pixels
# cv2.imshow('erosion',erosion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 2.Dilation ====================================
# dilation = cv2.dilate(img,kernel,iterations = 1) # iterations = 1, 左右個dilate pixels
# cv2.imshow('Dilation',dilation )
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 3.Opening is just another name of erosion followed by dilation.
# # It is useful in removing noise, as we explained above.
# img = cv2.imread('images/opening.png',0)
# print(img.shape)
# cv2.imshow('org.png',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# cv2.imshow('Opening',opening )
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 4.Closing is reverse of Opening, Dilation followed by Erosion.
# # It is useful in closing small holes inside the foreground objects,
# img = cv2.imread('images/closing.png',0)
# print(img.shape)
# cv2.imshow('org.png',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('closing',closing )
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 5. Morphological Gradient : The result will look like the outline of the object.
# # gradient : It is the difference between dilation and erosion of an image.
# # gradient is dilation - erosion
# img = cv2.imread('images/j.png',0)
# print(img.shape)
# cv2.imshow('j_png',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# cv2.imshow('gradient',gradient)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 6.Top Hat :It is the difference between input image and Opening of the image.
# img = cv2.imread('images/j.png',0)
# kernel = np.ones((5,5),np.uint8)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# tophot = img-opening
# t_ophot= cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# plt.subplot(221)
# plt.title("j image")
# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(222)
# plt.title("opening image")
# plt.imshow(opening, cmap='gray', vmin=0, vmax=255)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(223)
# plt.title("img-opening image")
# plt.imshow(tophot, cmap='gray', vmin=0, vmax=255)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(224)
# plt.title("MORPH_TOPHAT image")
# plt.imshow(t_ophot, cmap='gray', vmin=0, vmax=255)
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
# # 7. Black Hat It is the difference between the closing of the input image and input image.
# blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
# cv2.imshow('blackhat',blackhat)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 8 Structuring Element : elliptical/circular shaped kernels
#  cv.getStructuringElement().
# rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# print('rect_kernel=\n',rect_kernel)
# ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# print('ellipse_kernel=\n',ellipse_kernel)
# cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) # 十字
# print('cross_kernel=\n',cross_kernel)
