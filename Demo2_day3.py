# https://scipy.org/
# https://aws.amazon.com/tw/deepracer/

##########################
# use 彩色的方式讀單色     #
##########################
# demo3 print(image1[0,0]) # [1 1 1] b,g,r--------------
# import cv2
# import matplotlib
# from matplotlib import pyplot as plt
#
# matplotlib.rcParams['image.cmap'] = 'gray'
# IMAGE1 = 'images/number_zero.jpg'
# image1 = cv2.imread(IMAGE1,1) # 1 is color image
# print(type(image1), image1.shape)  # <class 'numpy.ndarray'> (13, 11, 3)
# cv2.imshow("from opencv", image1)  # 有show 但是很小注意看
# plt.imshow(image1)
# plt.show()
# print(image1.dtype) # uint8
# print(image1[0,0]) # [1 1 1] b,g,r

##(1) https://www.donkeycar.com/
## An opensource DIY self driving platform for small scale cars.
## RC CAR  +  Raspberry Pi + Python (tornado, keras, tensorflow, opencv, ....)

## (2) https://www.ros.org/
## that help you build robot applications. From drivers to
## state-of-the-art algorithms(最先進的), and with powerful developer tools,
## ROS has what you need for your next robotics project. And it's all open source.
## (3) https://www.lattepanda.com/ 迷你電腦買一台
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import cv2
# import matplotlib
# from matplotlib import pyplot as plt
#
# matplotlib.rcParams['image.cmap'] = 'gray'
# IMAGE1 = 'images/number_zero.jpg'
# image1 = cv2.imread(IMAGE1, 1)
# print(type(image1), image1.shape)
# cv2.imshow("from opencv", image1)
# plt.imshow(image1)
# plt.show()
# print(image1.dtype)
# print(image1[0, 0])
# plt.figure(figsize=[14, 6])
#
# image1[0, 0] = (255, 0, 0)
# image1[1, 1] = (0, 255, 0)
# image1[2, 2] = (0, 0, 255)
# plt.imshow(image1)
# plt.show()
#
# plt.subplot(1, 3, 1)
# plt.imshow(image1[:, :, 0])
# plt.subplot(1, 3, 2)
# plt.imshow(image1[:, :, 1])
# plt.subplot(1, 3, 3)
# plt.imshow(image1[:, :, 2])
# plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import cv2
# import matplotlib
# from matplotlib import pyplot as plt
#
# matplotlib.rcParams['image.cmap'] = 'gray'
# IMAGE1 = 'images/number_zero.jpg'
#
# image1 = cv2.imread(IMAGE1, 1)
# print(type(image1), image1.shape)
# cv2.imshow("from opencv", image1)
# plt.imshow(image1)
# plt.show()
#
# print(image1.dtype)
# print(image1[0, 0])
# plt.figure(figsize=[14, 6])
#
# image1[0, 0] = (255, 0, 0)
# image1[1, 1] = (0, 255, 0)
# image1[2, 2] = (0, 0, 255)
# # read as normal RGB
# plt.imshow(image1[:, :, ::-1])
# cv2.imshow("black as color channel", image1)
# plt.show()
#
# plt.subplot(1, 3, 1)
# cv2.imshow("first channel", image1[:, :, 0])
# plt.title("first channel")
# plt.imshow(image1[:, :, 0])
#
# plt.subplot(1, 3, 2)
# cv2.imshow("second channel", image1[:, :, 1])
# plt.title("second channel")
# plt.imshow(image1[:, :, 1])
#
# plt.subplot(1, 3, 3)
# cv2.imshow("third channel", image1[:, :, 2])
# plt.title("third channel")
# plt.imshow(image1[:, :, 2])
# plt.show()

# # demo4 -------------------------------------
# import cv2
# import matplotlib.pyplot as plt
#
# IMAGE1 = 'images/transparency1.png'
# image1 = cv2.imread(IMAGE1, -1)
# image2 = cv2.imread(IMAGE1, 0)
# image3 = cv2.imread(IMAGE1, 1)
#
# print(f"read as -1, shape={image1.shape}")  # read as -1, shape=(395, 386, 4) 第4通道是透明度
# print(f"read as 2, shape={image2.shape}")  #read as 2, shape=(395, 386)
# print(f"read as 3, shape={image3.shape}")  #read as 3, shape=(395, 386, 3)
#
# cv2.imshow("read as -1", image1)
# plt.imshow(image1)
# plt.show()
# cv2.imshow("read as 0", image2)
# plt.imshow(image2)
# plt.show()
# cv2.imshow("read as 1", image3)
# plt.imshow(image3)
# plt.show()

# image4 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)~~~~~~~~~~~~~~~~~~~
# import cv2
# import matplotlib.pyplot as plt
#
# IMAGE1 = 'images/transparency1.png'
# image1 = cv2.imread(IMAGE1, -1)
# image2 = cv2.imread(IMAGE1, 0)
# image3 = cv2.imread(IMAGE1, 1)
# print(f"read as -1, shape={image1.shape}")
# print(f"read as 2, shape={image2.shape}")
# print(f"read as 3, shape={image3.shape}")
# cv2.imshow("read as -1", image1)
# plt.imshow(image1)
# plt.show()
#
# cv2.imshow("read as 0", image2)
# plt.imshow(image2)
# plt.show()
# cv2.imshow("read as 1", image3)
# plt.imshow(image3)
# plt.show()
#
# image4 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# plt.title("correct with alpha channel(-1=keep)")
# plt.imshow(image4)
# plt.show()
# print(f"image4 with dimension=={image4.shape}")
#
# image5 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
# plt.title("correct without alpha(1, color)")
# plt.imshow(image5)
# plt.show()
# print(f"image5 with dimension=={image5.shape}")
#
# image6 = image1[:, :, :3]  # bgr
# image7 = image1[:, :, -1]  # rgb
# plt.figure(figsize=[12, 4])
# plt.subplot(131)
# plt.imshow(image6)
# plt.subplot(132)
# plt.imshow(image6[:, :, ::-1]) # 透明度
# plt.subplot(133)
# plt.imshow(image7)
# plt.show()

# ### demo5.py 畫直線 : ##################################################
# (1)line(image, point1, point2, color,[thickness,[linetype,[shift]]])
# (2)point1 & point2 is tuple(x,y)
# (3)color is 3個值的tuple (255,0,0) 注意顏色是BGR
# (4)line type is cv2.LINE_AA(Anti-Aliazing: 抗鋸齒)
# (5)左下角是原點
#########################################################################
# import cv2
# import matplotlib.pyplot as plt
#
# IMAGE1 = 'images/bg1.jpg'
# originalImage = cv2.imread(IMAGE1, -1)
# plt.imshow(originalImage[:, :, ::-1])
# plt.title("original image")
# plt.show()
#
# import cv2
# import matplotlib.pyplot as plt
#
# IMAGE1 = 'images/bg1.jpg'
# originalImage = cv2.imread(IMAGE1, -1)
# plt.imshow(originalImage[:, :, ::-1])
# plt.title("original image")
# plt.show()
#
# for i in range(0, 20):
#     cv2.line(originalImage, (200, 80 + 20 * i), (480, 80 + 10 * i),
#              (255 - 10 * i, 0 + 10 * i, 0), thickness=5, lineType=cv2.LINE_AA)
# # (255 - 10 * i, 0 + 10 * i, 0) : blue big to small, green small to big,red =0
# plt.imshow(originalImage[:, :, ::-1])
# plt.title("image with annotate")
# plt.show()

# 畫圓形 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# circle(image, center, radius, color, [thickness,[linetype,[shift]]])
# thickness 是圓邊緣ˇ的粗細，負值會填圖: = -1 填滿
##############################################################
# import cv2
# import matplotlib.pyplot as plt
#
# IMAGE1 = 'images/bg1.jpg'
# originalImage = cv2.imread(IMAGE1, -1)
# plt.imshow(originalImage[:, :, ::-1])
# plt.title("original image")
# plt.show()
#
# cv2.circle(originalImage, (250, 250), 100, (255, 255, 0),
#            thickness=10, lineType=cv2.LINE_AA)
# cv2.circle(originalImage, (150, 150), 100, (255, 0, 255),
#            thickness=-1, lineType=None)
# plt.imshow(originalImage[:, :, ::-1])
# plt.show()

# 橢圓 or 扇形#########################################
# ellipse(image, center,axes, angle, startAngle,endAngle,
#         color,[thickness,[linetype,[shift]]])
# center is tuple : (x,y) , axes is tuple : (長軸，短軸)
######################################################
# import cv2
# import matplotlib.pyplot as plt
#
# IMAGE1 = 'images/bg1.jpg'
# originalImage = cv2.imread(IMAGE1, -1)
# plt.imshow(originalImage[:, :, ::-1])
# plt.title("original image")
# plt.show()
#
# cv2.ellipse(originalImage, (400, 125), (200, 80), 0, 0, 360, (255, 0, 0), thickness=-1,
#             lineType=cv2.LINE_AA)
# cv2.ellipse(originalImage, (300, 125), (200, 80), 45, 0, 360, (0, 255, 0), thickness=5,
#             lineType=cv2.LINE_AA)
# cv2.ellipse(originalImage, (200, 125), (200, 80), 90, 0, 360, (0, 0, 255), thickness=7,
#             lineType=cv2.LINE_AA)
# plt.imshow(originalImage[:, :, ::-1])
# plt.title("annotated image")
# plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import cv2
# import matplotlib.pyplot as plt
#
# IMAGE1 = 'images/bg1.jpg'
# originalImage = cv2.imread(IMAGE1, -1)
# plt.imshow(originalImage[:, :, ::-1])
# plt.title("original image")
# plt.show()
#
# cv2.ellipse(originalImage, (400, 125), (200, 80), 0, 0, 360, (255, 0, 0), thickness=3,
#             lineType=cv2.LINE_AA)
# cv2.ellipse(originalImage, (300, 125), (200, 80), 45, 0, 360, (0, 255, 0), thickness=5,
#             lineType=cv2.LINE_AA)
# cv2.ellipse(originalImage, (200, 125), (200, 80), 90, 0, 360, (0, 0, 255), thickness=7,
#             lineType=cv2.LINE_AA)
# plt.imshow(originalImage[:, :, ::-1])
# plt.title("annotated image")
# plt.show()
#
# cv2.ellipse(originalImage, (700, 325), (200, 80), 0, 0, 90, (0, 0, 255), thickness=5,
#             lineType=cv2.LINE_AA)
# cv2.ellipse(originalImage, (700, 325), (80, 200), 0, 180, 270, (0, 255, 255), thickness=5,
#             lineType=cv2.LINE_AA)
# cv2.ellipse(originalImage, (700, 325), (140, 140), 0, 270, 360, (255, 255, 0), thickness=-1,
#             lineType=cv2.LINE_AA)
# plt.imshow(originalImage[:, :, ::-1])
# plt.title("annotated image")
# plt.show()

# demo8 文字 #######################################################
# putText(image, text, org, fontFace, fontScale,
#         color,[thickness,[linetype,[bottomleftOrigin]]])
# org is tuple:(50,50) 位置
####################################################################
import cv2
import matplotlib.pyplot as plt

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)
plt.imshow(originalImage[:, :, ::-1])
plt.title("original image")
plt.show()

text1 = "hello opencv in python"
cv2.putText(originalImage, text1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255),
            thickness=2, lineType=cv2.LINE_AA)
cv2.putText(originalImage, text1, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
            thickness=2, lineType=cv2.LINE_AA)
plt.imshow(originalImage[:, :, ::-1])
plt.show()

#############################################
# 使用 webcam/usb
# https://github.com/NVIDIA-AI-IOT/jetcam#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import cv2
#
# # you ==> 0
# # instructor ==>1
# capture = cv2.VideoCapture(1)
# while True:
#     returnValue, frame = capture.read()
#     cv2.imshow("this is me!", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     pass
# capture.release()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import cv2
#
# # you ==> 0
# # instructor ==>1
# capture = cv2.VideoCapture(1)
# while True:
#     returnValue, frame = capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("this is me!", gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     pass
# capture.release()
# cv2.destroyAllWindows()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# demo10
#
# import cv2
# import matplotlib.pyplot as plt
#
# IMAGE1 = 'images/bg1.jpg'
# originalImage = cv2.imread(IMAGE1, -1)
#
# k = 0
# while k != 27:
#     cv2.imshow("main window", originalImage)
#     cv2.putText(originalImage, "left click, and drag", (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
#     k = cv2.waitKey(20)
# cv2.destroyAllWindows()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import cv2
# import matplotlib.pyplot as plt
#
# IMAGE1 = 'images/bg1.jpg'
# originalImage = cv2.imread(IMAGE1, -1)
#
#
# def drawCircle(action, x, y, flags, userdata):
#     if action == cv2.EVENT_LBUTTONDOWN:
#         print("left bottom down, at({}.{})".format(x, y))
#     elif action == cv2.EVENT_LBUTTONUP:
#         print("left bottom release up, at({}.{})".format(x, y))
#
#
# cv2.namedWindow("main window")
# cv2.setMouseCallback("main window", drawCircle)
# k = 0
# while k != 27:
#     cv2.imshow("main window", originalImage)
#     cv2.putText(originalImage, "left click, and drag", (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
#     k = cv2.waitKey(20)
# cv2.destroyAllWindows()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import cv2
# import matplotlib.pyplot as plt
#
# IMAGE1 = 'images/bg1.jpg'
# originalImage = cv2.imread(IMAGE1, -1)
#
# center = [0, 0]
#
#
# def drawCircle(action, x, y, flags, userdata):
#     global center
#     if action == cv2.EVENT_LBUTTONDOWN:
#         center = (x, y)
#         print("left bottom down, at({}.{})".format(x, y))
#         cv2.circle(originalImage, center, 1, (255, 255, 0), 6, cv2.LINE_AA)
#     elif action == cv2.EVENT_LBUTTONUP:
#         print("left bottom release up, at({}.{})".format(x, y))
#
#
# WINDOW_NAME = "main window"
# cv2.namedWindow(WINDOW_NAME)
# cv2.setMouseCallback(WINDOW_NAME, drawCircle)
# k = 0
# while k != 27:
#     cv2.imshow(WINDOW_NAME, originalImage)
#     cv2.putText(originalImage, "left click, and drag", (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
#     k = cv2.waitKey(20)
# cv2.destroyAllWindows()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import cv2
# import matplotlib.pyplot as plt
#
# IMAGE1 = 'images/bg1.jpg'
# originalImage = cv2.imread(IMAGE1, -1)
# imageCopy = originalImage.copy()
# center = (0, 0)
#
#
# def drawCircle(action, x, y, flags, userdata):
#     global center
#     if action == cv2.EVENT_LBUTTONDOWN:
#         center = (x, y)
#         print("left bottom down, at({}.{})".format(x, y))
#         cv2.circle(originalImage, center, 1, (255, 255, 0), 6, cv2.LINE_AA)
#     elif action == cv2.EVENT_LBUTTONUP:
#         print("left bottom release up, at({}.{})".format(x, y))
#
#
# WINDOW_NAME = "main window"
# cv2.namedWindow(WINDOW_NAME)
# cv2.setMouseCallback(WINDOW_NAME, drawCircle)
# k = 0
# while k != 27:
#     cv2.imshow(WINDOW_NAME, originalImage)
#     cv2.putText(originalImage, "left click, and drag", (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
#     k = cv2.waitKey(20)
#     if k == ord('c'):
#         originalImage = imageCopy.copy()
# cv2.destroyAllWindows()
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# demo10
#
# import cv2
# import matplotlib.pyplot as plt
# import math
#
# IMAGE1 = 'images/bg1.jpg'
# originalImage = cv2.imread(IMAGE1, -1)
# imageCopy = originalImage.copy()
# center = (0, 0)
# bbox = (0, 0)
#
#
# def drawCircle(action, x, y, flags, userdata):
#     global center
#     if action == cv2.EVENT_LBUTTONDOWN:
#         center = (x, y)
#         print("left bottom down, at({}.{})".format(x, y))
#         cv2.circle(originalImage, center, 1, (255, 255, 0), 6, cv2.LINE_AA)
#     elif action == cv2.EVENT_LBUTTONUP:
#         bbox = (x, y)
#         radius = math.sqrt(math.pow(center[0] - bbox[0], 2) +
#                            math.pow(center[1] - bbox[1], 2))
#         cv2.circle(originalImage, center, int(radius), (0, 255, 255), 2, cv2.LINE_AA)
#         print("left bottom release up, at({}.{})".format(x, y))
#
#
# WINDOW_NAME = "main window"
# cv2.namedWindow(WINDOW_NAME)
# cv2.setMouseCallback(WINDOW_NAME, drawCircle)
# k = 0
# while k != 27:
#     cv2.imshow(WINDOW_NAME, originalImage)
#     cv2.putText(originalImage, "left click, and drag", (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
#     k = cv2.waitKey(20)
#     if k == ord('c'):
#         originalImage = imageCopy.copy()
# cv2.destroyAllWindows()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import cv2
# import matplotlib.pyplot as plt
# import math
#
# IMAGE1 = 'images/bg1.jpg'
# originalImage = cv2.imread(IMAGE1, -1)
# imageCopy = originalImage.copy()
# #center = (0, 0)
# #bbox = (0, 0)
#
#
# def drawCircle(action, x, y, flags, userdata):
#     global center
#     if action == cv2.EVENT_LBUTTONDOWN:
#         center = (x, y)
#         print("left bottom down, at({}.{})".format(x, y))
#         cv2.circle(originalImage, center, 1, (255, 255, 0), 6, cv2.LINE_AA)
#     elif action == cv2.EVENT_LBUTTONUP:
#         bbox = (x, y)
#         radius = math.sqrt(math.pow(center[0] - bbox[0], 2) +
#                            math.pow(center[1] - bbox[1], 2))
#         cv2.circle(originalImage, center, int(radius), (0, 255, 255), 2, cv2.LINE_AA)
#         print("left bottom release up, at({}.{})".format(x, y))
#
#
# WINDOW_NAME = "main window"
# cv2.namedWindow(WINDOW_NAME)
# cv2.setMouseCallback(WINDOW_NAME, drawCircle)
# k = 0
# while k != 27:
#     cv2.imshow(WINDOW_NAME, originalImage)
#     cv2.putText(originalImage, "left click, and drag", (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
#     k = cv2.waitKey(20)
#     if k == ord('c'):
#         originalImage = imageCopy.copy()
# cv2.destroyAllWindows()
#
#
# git config --list
# more .gitconfig
#
#
# pip install pillow
# https://www.google.com/get/noto/#sans-hant
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import cv2
# import matplotlib.pyplot as plt
# import math
# from PIL import ImageFont, ImageDraw, Image
# import numpy as np
#
# IMAGE1 = 'images/bg1.jpg'
# originalImage = cv2.imread(IMAGE1, -1)
# imageCopy = originalImage.copy()
#
#
# # center = (0, 0)
# # bbox = (0, 0)
#
#
# def drawCircle(action, x, y, flags, userdata):
#     global center
#     if action == cv2.EVENT_LBUTTONDOWN:
#         center = (x, y)
#         print("left bottom down, at({}.{})".format(x, y))
#         cv2.circle(originalImage, center, 1, (255, 255, 0), 6, cv2.LINE_AA)
#     elif action == cv2.EVENT_LBUTTONUP:
#         bbox = (x, y)
#         radius = math.sqrt(math.pow(center[0] - bbox[0], 2) +
#                            math.pow(center[1] - bbox[1], 2))
#         cv2.circle(originalImage, center, int(radius), (0, 255, 255), 2, cv2.LINE_AA)
#         print("left bottom release up, at({}.{})".format(x, y))
#
#
# WINDOW_NAME = "main window"
# cv2.namedWindow(WINDOW_NAME)
# cv2.setMouseCallback(WINDOW_NAME, drawCircle)
# FONT_PATH = 'fonts/NotoSerifCJK-Regular.ttc'
# font = ImageFont.truetype(FONT_PATH, 36)
# img_pil = Image.fromarray(originalImage)
# draw = ImageDraw.Draw(img_pil)
# k = 0
# while k != 27:
#     cv2.imshow(WINDOW_NAME, originalImage)
#     draw.text((10, 200), "按下左鍵left click, and drag", font=font, fill=(255, 255, 0, 255))
#     origImage = np.array(img_pil)
#     # cv2.putText(originalImage, "按下左鍵left click, and drag", (50, 50),
#     #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
#     k = cv2.waitKey(20)
#     if k == ord('c'):
#         originalImage = imageCopy.copy()
# cv2.destroyAllWindows()
# ~~~~~~~~~~~~~~~~~~~~~~~
# [fix1]
# import cv2
# import matplotlib.pyplot as plt
# import math
# from PIL import ImageFont, ImageDraw, Image
# import numpy as np
#
# IMAGE1 = 'images/bg1.jpg'
# originalImage = cv2.imread(IMAGE1, -1)
# imageCopy = originalImage.copy()
#
#
# # center = (0, 0)
# # bbox = (0, 0)
#
#
# def drawCircle(action, x, y, flags, userdata):
#     global center
#     if action == cv2.EVENT_LBUTTONDOWN:
#         center = (x, y)
#         print("left bottom down, at({}.{})".format(x, y))
#         cv2.circle(originalImage, center, 1, (255, 255, 0), 6, cv2.LINE_AA)
#     elif action == cv2.EVENT_LBUTTONUP:
#         bbox = (x, y)
#         radius = math.sqrt(math.pow(center[0] - bbox[0], 2) +
#                            math.pow(center[1] - bbox[1], 2))
#         cv2.circle(originalImage, center, int(radius), (0, 255, 255), 2, cv2.LINE_AA)
#         print("left bottom release up, at({}.{})".format(x, y))
#
#
# WINDOW_NAME = "main window"
# cv2.namedWindow(WINDOW_NAME)
# cv2.setMouseCallback(WINDOW_NAME, drawCircle)
# FONT_PATH = 'fonts/NotoSerifCJK-Regular.ttc'
# font = ImageFont.truetype(FONT_PATH, 36)
# img_pil = Image.fromarray(originalImage)
# draw = ImageDraw.Draw(img_pil)
# k = 0
# while k != 27:
#     cv2.imshow(WINDOW_NAME, originalImage)
#     draw.text((10, 200), "按下左鍵left click, and drag", font=font, fill=(255, 255, 0, 255))
#     originalImage = np.array(img_pil)
#     # cv2.putText(originalImage, "按下左鍵left click, and drag", (50, 50),
#     #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
#     k = cv2.waitKey(20)
#     if k == ord('c'):
#         originalImage = imageCopy.copy()
# cv2.destroyAllWindows()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# import cv2
# import matplotlib.pyplot as plt
# import math
# from PIL import ImageFont, ImageDraw, Image
# import numpy as np
#
# IMAGE1 = 'images/bg1.jpg'
# originalImage = cv2.imread(IMAGE1, -1)
# imageCopy = originalImage.copy()
#
#
# # center = (0, 0)
# # bbox = (0, 0)
#
#
# def drawCircle(action, x, y, flags, userdata):
#     global center
#     if action == cv2.EVENT_LBUTTONDOWN:
#         center = (x, y)
#         print("left bottom down, at({}.{})".format(x, y))
#         cv2.circle(originalImage, center, 1, (255, 255, 0), 6, cv2.LINE_AA)
#     elif action == cv2.EVENT_LBUTTONUP:
#         bbox = (x, y)
#         radius = math.sqrt(math.pow(center[0] - bbox[0], 2) +
#                            math.pow(center[1] - bbox[1], 2))
#         cv2.circle(originalImage, center, int(radius), (0, 255, 255), 2, cv2.LINE_AA)
#         print("left bottom release up, at({}.{})".format(x, y))
#
#
# WINDOW_NAME = "main window"
# cv2.namedWindow(WINDOW_NAME)
# cv2.setMouseCallback(WINDOW_NAME, drawCircle)
# FONT_PATH = 'fonts/NotoSerifCJK-Regular.ttc'
# font = ImageFont.truetype(FONT_PATH, 36)
# k = 0
# while k != 27:
#     cv2.imshow(WINDOW_NAME, originalImage)
#     img_pil = Image.fromarray(originalImage)
#     draw = ImageDraw.Draw(img_pil)
#     draw.text((10, 200), "按下左鍵left click, and drag", font=font, fill=(255, 255, 0, 255))
#     originalImage = np.array(img_pil)
#     # cv2.putText(originalImage, "按下左鍵left click, and drag", (50, 50),
#     #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
#     k = cv2.waitKey(20)
#     if k == ord('c'):
#         originalImage = imageCopy.copy()
# cv2.destroyAllWindows()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# pip install flask imutils
#
# make a directory demo11
#
# make a sub directory templates
#
#
#
# demo11.py
#
# from flask import Flask, render_template, Response
#
# app = Flask(__name__)
#
# if __name__ == '__main__':
#     app.run(debug=True, threaded=True, use_reloader=False)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# index.html
#
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <title>AIOCV Lab11</title>
# </head>
# <body>
# <H1>Open CV integrate with Flask</H1>
# <img src="{{url_for('video_feed')}}"/>
# </body>
# </html>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# demo11.py
#
# from flask import Flask, render_template, Response
#
# app = Flask(__name__)
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
# if __name__ == '__main__':
#     app.run(debug=True, threaded=True, use_reloader=False)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# demo11.py
#
# from flask import Flask, render_template, Response
# import cv2
#
# app = Flask(__name__)
#
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
#
# def generate():
#     global outputFrame, lock
#     while True:
#         with lock:
#             if outputFrame is None:
#                 continue
#             flag, encodedImage = cv2.imencode("*.jpg", outputFrame)
#             if not flag:
#                 continue
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage)
#                + b'\r\n')
#
#
# @app.route("/video_feed")
# def video_feed():
#     return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
#
#
# if __name__ == '__main__':
#     app.run(debug=True, threaded=True, use_reloader=False)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# from flask import Flask, render_template, Response
# import cv2
# import threading
# from imutils.video import VideoStream
# import imutils
# import datetime
#
# app = Flask(__name__)
#
# outputFrame = None
# lock = threading.Lock()
# # default == 0
# # instructor ==1
# vs = VideoStream(src=1).start()
#
#
# def getScreen(frameCount):
#     global vs, outputFrame, lock
#     total = 0
#     while True:
#         frame = vs.read()
#         frame = imutils.resize(frame, width=800)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         timestamp = datetime.datetime.now()
#         cv2.putText(gray, timestamp.strftime("%A %d %B %Y %I:%M:%S:%p"),
#                     (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.35, (255, 0, 255), 1)
#         total += 1
#         with lock:
#             outputFrame = gray.copy()
#
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
#
# def generate():
#     global outputFrame, lock
#     while True:
#         with lock:
#             if outputFrame is None:
#                 continue
#             flag, encodedImage = cv2.imencode(".jpg", outputFrame)
#             if not flag:
#                 continue
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage)
#                + b'\r\n')
#
#
# @app.route("/video_feed")
# def video_feed():
#     return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
#
#
# if __name__ == '__main__':
#     t = threading.Thread(target=getScreen, args=(32,))
#     t.daemon = True
#     t.start()
#     app.run(debug=True, threaded=True, use_reloader=False)
# vs.stop()
