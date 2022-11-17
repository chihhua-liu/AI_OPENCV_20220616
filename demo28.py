## demo28

# import cv2
# import matplotlib
# from matplotlib import pyplot as plt
#
# matplotlib.rcParams['figure.figsize'] = (4.0, 4.0)
# matplotlib.rcParams['image.cmap'] = 'gray'
# FILENAME = 'images/morph1.jpg'
# image = cv2.imread(FILENAME)
# plt.imshow(image)
# # plt.show()
# # 3,3, 5,5 , 77, 99
# ksize = (9, 9)
# kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
# plt.imshow(kernel1)
# # plt.show()
# # cv2.MORPH_OPEN
# morphedImage = image.copy()
# for _ in range(20):
#     morphedImage = cv2.morphologyEx(morphedImage, cv2.MORPH_CLOSE, kernel1)
#
# plt.figure(figsize=[15, 9])
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.subplot(1, 2, 2)
# plt.imshow(morphedImage)
# plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'
FILENAME = 'images/mickey.jpg'
image = cv2.imread(FILENAME)
plt.imshow(image)

ksize = (5, 5)
k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
plt.imshow(k1)

morphedImage = image.copy()
morph1 = cv2.morphologyEx(morphedImage, cv2.MORPH_ERODE, k1)
morph2 = cv2.morphologyEx(morphedImage, cv2.MORPH_DILATE, k1)
morph3 = cv2.morphologyEx(morphedImage, cv2.MORPH_GRADIENT, k1)

plt.figure(figsize=[12,6])
plt.subplot(141)
plt.title("original")
plt.imshow(image)
plt.subplot(142)
plt.title("erode")
plt.imshow(morph1)
plt.subplot(143)
plt.title("dilate")
plt.imshow(morph2)
plt.subplot(144)
plt.title("gradient")
plt.imshow(morph3)
plt.show()
