https://scipy.org/
https://aws.amazon.com/tw/deepracer/

#########################
# use 彩色的方式讀單色     #
#########################
demo3 print(image1[0,0]) # [1 1 1] b,g,r--------------
import cv2
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['image.cmap'] = 'gray'
IMAGE1 = 'images/number_zero.jpg'
image1 = cv2.imread(IMAGE1,1) # 1 is color image
print(type(image1), image1.shape)  # <class 'numpy.ndarray'> (13, 11, 3)
cv2.imshow("from opencv", image1)  # 有show 但是很小注意看
plt.imshow(image1)
plt.show()
print(image1.dtype) # uint8
print(image1[0,0]) # [1 1 1] b,g,r

#(1) https://www.donkeycar.com/
# An opensource DIY self driving platform for small scale cars.
# RC CAR  +  Raspberry Pi + Python (tornado, keras, tensorflow, opencv, ....)

# (2) https://www.ros.org/
# that help you build robot applications. From drivers to
# state-of-the-art algorithms(最先進的), and with powerful developer tools,
# ROS has what you need for your next robotics project. And it's all open source.
# (3) https://www.lattepanda.com/ 迷你電腦買一台
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['image.cmap'] = 'gray'
IMAGE1 = 'images/number_zero.jpg'
image1 = cv2.imread(IMAGE1, 1)
print(type(image1), image1.shape)
cv2.imshow("from opencv", image1)
plt.imshow(image1)
plt.show()
print(image1.dtype)
print(image1[0, 0])
plt.figure(figsize=[14, 6])

image1[0, 0] = (255, 0, 0)
image1[1, 1] = (0, 255, 0)
image1[2, 2] = (0, 0, 255)
plt.imshow(image1)
plt.show()

plt.subplot(1, 3, 1)
plt.imshow(image1[:, :, 0])
plt.subplot(1, 3, 2)
plt.imshow(image1[:, :, 1])
plt.subplot(1, 3, 3)
plt.imshow(image1[:, :, 2])
plt.show()

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['image.cmap'] = 'gray'
IMAGE1 = 'images/baby.jpg'

image1 = cv2.imread(IMAGE1, 1)
print(type(image1), image1.shape)
cv2.imshow("from opencv", image1)
plt.imshow(image1)
plt.show()

print(image1.dtype)
print(image1[0, 0])
plt.figure(figsize=[14, 6])

image1[0, 0] = (255, 0, 0)
image1[1, 1] = (0, 255, 0)
image1[2, 2] = (0, 0, 255)
# read as normal RGB
plt.imshow(image1[:, :, ::-1])
cv2.imshow("black as color channel", image1)
plt.show()

plt.subplot(1, 3, 1)
cv2.imshow("first channel", image1[:, :, 0])
plt.title("first channel")
plt.imshow(image1[:, :, 0])

plt.subplot(1, 3, 2)
cv2.imshow("second channel", image1[:, :, 1])
plt.title("second channel")
plt.imshow(image1[:, :, 1])

plt.subplot(1, 3, 3)
cv2.imshow("third channel", image1[:, :, 2])
plt.title("third channel")
plt.imshow(image1[:, :, 2])
plt.show()

# demo4 -------------------------------------
import cv2
import matplotlib.pyplot as plt

IMAGE1 = 'images/transparency1.png'
image1 = cv2.imread(IMAGE1, -1)
image2 = cv2.imread(IMAGE1, 0)
image3 = cv2.imread(IMAGE1, 1)

print(f"read as -1, shape={image1.shape}")  # read as -1, shape=(395, 386, 4) 第4通道是透明度
print(f"read as 2, shape={image2.shape}")  #read as 2, shape=(395, 386)
print(f"read as 3, shape={image3.shape}")  #read as 3, shape=(395, 386, 3)

cv2.imshow("read as -1", image1)
plt.imshow(image1)
plt.show()
cv2.imshow("read as 0", image2)
plt.imshow(image2)
plt.show()
cv2.imshow("read as 1", image3)
plt.imshow(image3)
plt.show()

image4 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)~~~~~~~~~~~~~~~~~~~
import cv2
import matplotlib.pyplot as plt

IMAGE1 = 'images/transparency1.png'
image1 = cv2.imread(IMAGE1, -1)
image2 = cv2.imread(IMAGE1, 0)
image3 = cv2.imread(IMAGE1, 1)
print(f"read as -1, shape={image1.shape}")
print(f"read as 2, shape={image2.shape}")
print(f"read as 3, shape={image3.shape}")
cv2.imshow("read as -1", image1)
plt.imshow(image1)
plt.show()

cv2.imshow("read as 0", image2)
plt.imshow(image2)
plt.show()
cv2.imshow("read as 1", image3)
plt.imshow(image3)
plt.show()

image4 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
plt.title("correct with alpha channel(-1=keep)")
plt.imshow(image4)
plt.show()
print(f"image4 with dimension=={image4.shape}")

image5 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
plt.title("correct without alpha(1, color)")
plt.imshow(image5)
plt.show()
print(f"image5 with dimension=={image5.shape}")

image6 = image1[:, :, :3]  # bgr
image7 = image1[:, :, -1]  # rgb
plt.figure(figsize=[12, 4])
plt.subplot(131)
plt.imshow(image6)
plt.subplot(132)
plt.imshow(image6[:, :, ::-1]) # 透明度
plt.subplot(133)
plt.imshow(image7)
plt.show()
----------------------------------------------------------------
drawing image:
cv.line , cv.rectangle ,cv.circle , cv.ellipse, cv.polylines,cv.putText
import numpy as np
import cv2 as cv

# Create a black image
img = np.zeros((512,512,3), np.uint8)
cv.imshow("np_zerosImage",img)
cv.waitKey(0)
cv.destroyAllWindows()

# Draw a diagonal blue line with thickness of 5 px
cv.line(img,(0,0),(511,511),(255,0,0),5)

# Draw rectangle
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)

# Draw circle
cv.circle(img,(447,63), 63, (0,0,255), -1)

# Draw Ellipse
cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

# Draw polygon
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((4,1,2))
cv.polylines(img,[pts],True,(0,255,255))

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,"OpenCV",(10,500),font, 2,(255,255,255),4,cv.LINE_AA) #LINE_AA is 實線
# Show image
cv.imshow('Drawing',img)
cv.waitKey(0)
cv.destroyAllWindows()
----------------------------------------------------------
#############################-------------------------------------------
cv.resize() : resize 縮放圖片#
#############################
interpolation	說明 : (1) INTER_NEAREST	最近鄰插值 (2)NTER_LINEAR	雙線性插值（預設）
(3) INTER_AREA: 用像素區域關係進行重採樣。它可能是圖像抽取的首選方法，因為它會產生無雲紋理(波紋)的結果。 但是當圖像縮放時，它類似於INTER_NEAREST方法。
(4) INTER_CUBIC	4x4像素鄰域的雙三次插值   (5)NTER_LANCZOS4	8x8像素鄰域的Lanczos插值
import numpy as np
import cv2 as cv

# Load an color image in grayscale
img = cv.imread('images/mybaby.jpg',0)
# zoom image
height, width = img.shape[:2]
res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)
cv.imshow('zoom image',res)

# shrink image, using other parameters
res = cv.resize(img,None,fx=0.5, fy=0.5, interpolation = cv.INTER_AREA)
cv.imshow('shrink image',res)

cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('otherbaby.jpg',res)
-------------------------------------------------------------
補充:
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('no argument')
        sys.exit()
    print('hello')
    print('sys.argv[0]=',sys.argv[0])  # C:\Users\mikal\PycharmProjects\AI_OPENCV_20220616\Demo2_day3.py
    print('sys.argv[1]=',sys.argv[1])  # 1234
-------------------------------------------------------------
in CMD : cd  C:\Users\mikal\PycharmProjects\AI_OPENCV_20220616>
then: Run demo2_day3.py test -> get: test
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument('string', type=str)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(args.string)
    print(type(args.string))
--------------------------------------------------------
class Demo:
    def __init__(self, i):
        self.i = i
        self.x = "xxx"
        self.y = "yyy"
        self.z = "zzz"

    def __str__(self):
        return str(self.i)

    def hello(self):
        print("hello " + self.__str__())


class SubDemo(Demo):
    pass


print('vars(Demo(22)=',vars(Demo(22))) #回傳 object 的屬性與屬性值的字典
print('vars(SubDemo)=',vars(SubDemo))
print('vars()=',vars())
############
色彩空間轉換 #  --------------------------------------
############
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,help = "Path to the image",default = "C:/Users/mikal/PycharmProjects/AI_OPENCV_20220616/images/baby.jpg")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
File_name ='images/baby.jpg'
image =cv2.imread(File_name)
cv2.imshow("Original", image)

# 206 types of cvtColor
# Gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

# Hue, Saturation, Value
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)

# Hue, Lightness/Luminance, Saturation
lab = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
cv2.imshow("HLS", lab)

# Lightness, A(Green..Red), B(Blue..Yellow)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)
cv2.waitKey(0)

### demo5.py 畫直線 : ##################################################
(1)line(image, point1, point2, color,[thickness,[linetype,[shift]]])
(2)point1 & point2 is tuple(x,y)
(3)color is 3個值的tuple (255,0,0) 注意顏色是BGR
(4)line type is cv2.LINE_AA(Anti-Aliazing: 抗鋸齒)
(5)左下角是原點
########################################################################
import cv2
import matplotlib.pyplot as plt

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)
plt.imshow(originalImage[:, :, ::-1])
plt.title("original image")
plt.show()

import cv2
import matplotlib.pyplot as plt

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)
plt.imshow(originalImage[:, :, ::-1])
plt.title("original image")
plt.show()

for i in range(0, 20):
    cv2.line(originalImage, (200, 80 + 20 * i), (480, 80 + 10 * i),
             (255 - 10 * i, 0 + 10 * i, 0), thickness=5, lineType=cv2.LINE_AA)
# (255 - 10 * i, 0 + 10 * i, 0) : blue big to small, green small to big,red =0
plt.imshow(originalImage[:, :, ::-1])
plt.title("image with annotate")
plt.show()

畫圓形 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
circle(image, center, radius, color, [thickness,[linetype,[shift]]])
thickness 是圓邊緣ˇ的粗細，負值會填圖: = -1 填滿
#############################################################
import cv2
import matplotlib.pyplot as plt

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)
plt.imshow(originalImage[:, :, ::-1])
plt.title("original image")
plt.show()

cv2.circle(originalImage, (250, 250), 100, (255, 255, 0),
           thickness=10, lineType=cv2.LINE_AA)
cv2.circle(originalImage, (150, 150), 100, (255, 0, 255),
           thickness=-1, lineType=None)
plt.imshow(originalImage[:, :, ::-1])
plt.show()

橢圓 or 扇形#########################################
ellipse(image, center,axes, angle, startAngle,endAngle,
        color,[thickness,[linetype,[shift]]])
center is tuple : (x,y) , axes is tuple : (長軸，短軸)
#####################################################
import cv2
import matplotlib.pyplot as plt

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)
plt.imshow(originalImage[:, :, ::-1])
plt.title("original image")
plt.show()

cv2.ellipse(originalImage, (400, 125), (200, 80), 0, 0, 360, (255, 0, 0), thickness=-1,
            lineType=cv2.LINE_AA)
cv2.ellipse(originalImage, (300, 125), (200, 80), 45, 0, 360, (0, 255, 0), thickness=5,
            lineType=cv2.LINE_AA)
cv2.ellipse(originalImage, (200, 125), (200, 80), 90, 0, 360, (0, 0, 255), thickness=7,
            lineType=cv2.LINE_AA)
plt.imshow(originalImage[:, :, ::-1])
plt.title("annotated image")
plt.show()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 cv2.ellipse #
###############
import cv2
import matplotlib.pyplot as plt

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)
plt.imshow(originalImage[:, :, ::-1])
plt.title("original image")
plt.show()

cv2.ellipse(originalImage, (400, 125), (200, 80), 0, 0, 360, (255, 0, 0), thickness=3,
            lineType=cv2.LINE_AA)
cv2.ellipse(originalImage, (300, 125), (200, 80), 45, 0, 360, (0, 255, 0), thickness=5,
            lineType=cv2.LINE_AA)
cv2.ellipse(originalImage, (200, 125), (200, 80), 90, 0, 360, (0, 0, 255), thickness=7,
            lineType=cv2.LINE_AA)
plt.imshow(originalImage[:, :, ::-1])
plt.title("annotated image")
plt.show()

cv2.ellipse(originalImage, (700, 325), (200, 80), 0, 0, 90, (0, 0, 255), thickness=5,
            lineType=cv2.LINE_AA)
cv2.ellipse(originalImage, (700, 325), (80, 200), 0, 180, 270, (0, 255, 255), thickness=5,
            lineType=cv2.LINE_AA)
cv2.ellipse(originalImage, (700, 325), (140, 140), 0, 270, 360, (255, 255, 0), thickness=-1,
            lineType=cv2.LINE_AA)
plt.imshow(originalImage[:, :, ::-1])
plt.title("annotated image")
plt.show()

demo8 文字 #######################################################
putText(image, text, org, fontFace, fontScale,
        color,[thickness,[linetype,[bottomleftOrigin]]])
org is tuple:(50,50) 位置
###################################################################
image 寫上文字 #
###############
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
------------------------------------------
補充 : (1)chr(65) # Convert integer 65 to ASCII Character ('A')
y = chr(65)
print(type(y), y)

# Print A-Z
for i in range(65, 65+25):
    print(chr(i), end = " , ")

(2)ord('A') # Convert ASCII Unicode Character 'A' to 65
y = ord('q')
print(type(y), y)

alphabet_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Print 65-90
for i in alphabet_list:
    print(ord(i), end = " , ")
------------------------------------------------
shift Image : #
############### M = np.float32([[1,0,100],[0,1,50]])   # 100,50 是 位移的x,y
               dst = cv.warpAffine(img,M,(cols,rows))  # (cols,rows) 輸出影像的大小
import numpy as np
import cv2 as cv

# Load an color image in grayscale
img = cv.imread('images/mybaby.jpg',cv.IMREAD_GRAYSCALE)

rows,cols = img.shape
M = np.float32([[1,0,100],[0,1,50]])   # 100,50 是 位移的x,y
dst = cv.warpAffine(img,M,(cols,rows))  # (cols,rows) 輸出影像的大小

cv.imshow('translation image',dst)
cv.waitKey(0)
cv.destroyAllWindows()
--------------------------------------------------------
 image rotate #
#################
import numpy as np
import cv2 as cv

# Load an color image in grayscale
img = cv.imread('images/mybaby.jpg',cv.IMREAD_GRAYSCALE)
cv.imshow('org image',img)
rows,cols = img.shape

# cols-1 and rows-1 are the coordinate limits (任一角度旋轉)
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),60,2) # 1: 不放大， 0.5 縮小， 2:放大
dst = cv.warpAffine(img,M,(cols*2,rows*2))

cv.imshow('Rotate image',dst)
cv.waitKey(0)
cv.destroyAllWindows()
# cv.rotate 可以旋轉3個方向----------------------------------
img90 = cv.rotate(img,cv.ROTATE_90_CLOCKWISE) #順時針 90
img180 = cv.rotate(img,cv.ROTATE_180)   ##順時針 180
img270 = cv.rotate(img,cv.ROTATE_90_COUNTERCLOCKWISE) #逆時針90度

cv.imshow('Rotate 90',img90)
cv.imshow('Rotate 180',img180)
cv.imshow('Rotate 270',img270)

cv.waitKey(0)
cv.destroyAllWindows()
--------------------------------------------------
image 翻轉 : cv.flip #
######################
import numpy as np
import cv2 as cv

image = cv.imread('images/mybaby.jpg')
cv.imshow("Original", image)

flipped = cv.flip(image, 0) # 0 is 沿著x 軸翻轉# 1 is 沿著y 軸翻轉 # -1 沿著兩軸翻轉
cv.imshow("Flipped Horizontally:沿著x 軸翻轉", flipped)
# cv.waitKey(0)
cv.destroyAllWindows()
----------------------------------------------
image 截圖 : ##
############## OpenCV 沒有截圖finction 用numpy slicing: mage[0:190, 125:300]
import numpy as np
import cv2 as cv

image = cv.imread('images/mybaby.jpg')
cv.imshow("Original", image)
# cropped = image[0:190, 125:300]
cv.imshow("Baby Face", cropped)
cv.waitKey(0)
---------------------------------------------
image 位元運算 #
###############
cv2.bitwise_and,bitwiseOr,cv2.bitwise_or, bitwiseXor, v2.bitwise_not
import numpy as np
import cv2

# Draw filled rectangle
rectangle = np.zeros((300, 300), dtype = "uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1) # -1 實心
cv2.imshow("Rectangle", rectangle)

# Draw filled circle
circle = np.zeros((300, 300), dtype = "uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("Circle", circle)
cv2.waitKey(0)

# and expression
bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAnd)
cv2.waitKey(0)

# or expression
bitwiseOr = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR", bitwiseOr)
cv2.waitKey(0)

# xor expression
bitwiseXor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR", bitwiseXor)
cv2.waitKey(0)

# not expression
bitwiseNot = cv2.bitwise_not(circle)
cv2.imshow("NOT", bitwiseNot)
cv2.waitKey(0)
cv2.destroyAllWindows()
--------------------------------------------------
iamge Masking #
################
import numpy as np
import argparse
import cv2
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True,
#     help = "Path to the image")
# args = vars(ap.parse_args())
#image = cv2.imread(args["image"])
image = cv2.imread('images/baby.jpg')
cv2.imshow("Original", image)
# draw white rectangle on the center of the image
mask = np.zeros(image.shape[:2], dtype = "uint8")
print(f'image width ={image.shape[1]}, image height = {image.shape[0]}')
print(f'cx = {image.shape[1]//2}, cy = {image.shape[0]//2}')

(cX, cY) = (image.shape[1]//2, image.shape[0]//2)   # 整數除法
cv2.rectangle(mask, (cX-100,cY-100), (cX+100,cY+100), 255, -1)
cv2.imshow("Mask", mask)

# masking image
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)

# draw white circle on the center of the image
mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.circle(mask, (cX, cY), 100, 255, -1)

# masking image
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask", mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)
cv2.destroyAllWindows()
-----------------------------------------------
image 模糊/平滑 #
################
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,  # "-i " is 小名
    help = "Path to the image")                   # 指令:python demo2_day3.py --image images/baby.jpg
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
# image = cv2.imread('images/baby.jpg')

cv2.imshow("Original", image)

# stack output images together: hstack & vstack
blurred = np.hstack([
    cv2.blur(image, (3, 3)),  # 取平均值當kernel value
    cv2.blur(image, (5, 5)),
    cv2.blur(image, (7, 7))])

cv2.imshow("Averaged", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
--------------------------------------------------
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# stack output images together
blurred = np.hstack([
    cv2.medianBlur(image, 3),   # 取中位數當當kernel value
    cv2.medianBlur(image, 5),
    cv2.medianBlur(image, 7)])

cv2.imshow("Median", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
#----------------------------------------------------
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image")   # python demo2_day3.py --image images/baby.jpg
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# stack output images together
blurred = np.hstack([
    cv2.GaussianBlur(image, (3, 3), 0),  #高斯加權平均值
    cv2.GaussianBlur(image, (5, 5), 0),
    cv2.GaussianBlur(image, (7, 7), 0)])

cv2.imshow("Gaussian", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
--------------------------------------------
雙邊濾波模糊 :cv2.bilateralFilter #  [baɪˋlætərəl] 左右對稱的；雙方的；雙邊的
#################################
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# stack output images together
blurred = np.hstack([
    cv2.bilateralFilter(image, 5, 21, 21),    # 5 is kernel
    cv2.bilateralFilter(image, 7, 31, 31),
    cv2.bilateralFilter(image, 9, 41, 41)])

cv2.imshow("Bilateral", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
-----------------------------------------
image 銳化 sharpening : cv2.filter2D() #
########################################
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
cv2.imshow('Original', img)

# generating the kernels

kernel_sharp1 = np.array([[0,-1,0],
                          [-1,5,-1],
                          [0,-1,0]])

kernel_sharp2 = np.array([[-1,-1,-1],
                          [-1,9,-1],
                          [-1,-1,-1]])

kernel_sharp3 = np.array([[1,1,1],
                          [1,-7,1],
                          [1,1,1]])

kernel_sharp4 = np.array([[-1,-1,-1,-1,-1],
                          [-1,2,2,2,-1],
                          [-1,2,8,2,-1],
                          [-1,2,2,2,-1],
                          [-1,-1,-1,-1,-1]]) / 8.0

# applying different kernels to the input image
out1 = cv2.filter2D(img, -1, kernel_sharp1)  # -1 表示維持原圖色彩
out2 = cv2.filter2D(img, -1, kernel_sharp2)
out3 = cv2.filter2D(img, -1, kernel_sharp3)
out4 = cv2.filter2D(img, -1, kernel_sharp4)

cv2.imshow('Sharpening', out1)
cv2.imshow('More Sharpening', out2)
cv2.imshow('Excessive Sharpening', out3)
cv2.imshow('Edge Enhancement', out4)
cv2.waitKey(0)
cv2.destroyAllWindows()
#------------------------------------------
Threshold #
############
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)  # 先 Blur 再 threshold，效果比較好
cv2.imshow("Image", image)

(T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)  # T is 155，thresh is image
cv2.imshow("Threshold Binary", thresh)

(T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Binary Inverse", threshInv)

cv2.imshow("Coins", cv2.bitwise_and(image, image,mask=thresh)) #mask=threshInv))
cv2.waitKey(0)
cv2.destroyAllWindows()
-----------------------------------------------------------------
threshold_adaptive(自適應臨界值) #
################################
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image)

thresh = cv2.adaptiveThreshold(blurred, 255,
cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
cv2.imshow("Mean Thresh", thresh)

thresh = cv2.adaptiveThreshold(blurred, 255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow("Gaussian Thresh", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
----------------------------------------------------
Otsu Threshold #
#################
using cv2 module
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image)

# Otsu threshold
th2,img2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
print("Otsu threshold: {}".format(th2))
cv2.imshow("Otsu", img2)

# Otsu's thresholding after Gaussian filtering
th3,img3 = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
print("Otsu threshold after Gaussian blur: {}".format(th3))
cv2.imshow("Otsu blur", img3)

cv2.waitKey(0)
cv2.destroyAllWindows()

Mahotas 是計算機視覺和影象處理 Python 庫。它包含大量影象處理演算法
包含演算法: 1.分水嶺 2.凸點計算。 3.擊中/擊不中,細化演算法。,4.澤尼克&;Haralick,枸杞多糖,和TAS的功能。
5.基於freeimage的numpy影象載入(需要安裝freeimage庫)。
6. 加速的魯棒特徵(SURF)等。7.閾值。8 卷積。 9 Sobel邊緣檢測。10 多邊形繪製
11. 距離變換 12.特徵計算  13.樣條插值
--------------------------------------------------
need mahotas module
import numpy as np
import argparse
import mahotas
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image)

# Otsu threshold
T = mahotas.thresholding.otsu(blurred)
print("Otsu? threshold: {}".format(T))
print('T=',T)

thresh = image.copy()
thresh[thresh > T] = 255
thresh[thresh < 255] = 0
cv2.imshow("T threshold", thresh)
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Otsu", thresh)

# Riddler-Calvard threshold
T = mahotas.thresholding.rc(blurred)
print("Riddler-Calvard: {}".format(T))
thresh = image.copy()
thresh[thresh > T] = 255
thresh[thresh < 255] = 0
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Riddler-Calvard", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
---------------------------------------------
Eage canny 邊緣偵測 #
####################
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Blurred", image)

# Canny edge detection
canny = cv2.Canny(image, 30, 150)
cv2.imshow("Canny", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
-------------------------------------------------
edge_sobel_laplacian #
#######################
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)

# Laplacian edge detection
lap = cv2.Laplacian(image, cv2.CV_64F)
#lap = np.uint8(np.absolute(lap))
lap = cv2.convertScaleAbs(lap)
cv2.imshow("Laplacian", lap)
cv2.waitKey(0)

# Sobel edge detection
sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

#sobelX = np.uint8(np.absolute(sobelX))
#sobelY = np.uint8(np.absolute(sobelY))
sobelX = cv2.convertScaleAbs(sobelX)
sobelY = cv2.convertScaleAbs(sobelY)
#sobelCombined = cv2.bitwise_or(sobelX, sobelY)
sobelCombined = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)

cv2.imshow("Sobel X", sobelX)
cv2.imshow("Sobel Y", sobelY)
cv2.imshow("Sobel Combined", sobelCombined)
cv2.waitKey(0)
cv2.destroyAllWindows()
---------------------------------------------------
contours : 先用 Threshold & candy Preprocessing #
#################################################
cv2.findContours(image, mode ,method)
mode: 1.cv2.RETR_EXTERNAL:取最外的輪廓 ， 2.cv2.RETR_LIST 取全部
method : 1常用:CHAIN_APPROX_SIMPLE : 會壓縮水平，垂直與對角線，只保留端點
         2 CHAIN_APPROX_NONE : 取所有座標，浪費資源
劃出contours : use cv2.drawContours(image, countours,comtouridx(第幾個contours， -1 is all),color,lineW)
劃出 blob ROI: cv2.boundingRect
import numpy as np
import argparse
import cv2
import d2l
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image")

args = vars(ap.parse_args()) # CMD command : python demo2_day3.py --image images/mickey.jpg

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)   #邊緣偵測時，通常會先做平滑化(cv2.GaussianBlur)來降低雜訊
#blurred = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imshow("Image", image)
#  # Canny is 邊緣運算(Edge Detection)
#  30 is 最小門檻值 threshold1 小於30 is 非邊緣
#  30~150 之間:分類為邊緣或非邊緣
#  >150 is 邊緣
edged = cv2.Canny(blurred, 30, 150)
cv2.imshow("Edges", edged)

(cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_LIST,#cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)

print("I count {} objects in this image".format(len(cnts)))

objs = image.copy()
cv2.drawContours(objs, cnts, -1, (0, 255, 0), 2)

bounding_boxes = [cv2.boundingRect(cnt) for cnt in cnts]
for bbox in bounding_boxes:
    [x,y,w,h]=bbox
    cv2.rectangle(objs,(x,y),(x+w,y+h),(0,255,0),2)
# fig = d2l.plt.imshow(image, cmap='gray')
# for i in range(0,len(cnts)):
#     x,y,w,h =cv2.boundingRect(cnts[i])
#     fig.axes.add_patch(d2l.plt.Rectangle((x, y), w, h, fill=False,
#                                          linestyle="-", edgecolor=color[i],
#                                          linewidth=2))
cv2.imshow("Objects", objs)
cv2.waitKey(0)
cv2.destroyAllWindows()
-------------------------------------------------------
erosion_dilation  #
####################
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1] # [0] is 255, [1] is image
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

# we apply erosions to reduce the size of foreground objects
mask = thresh.copy()
kernel = np.ones((5,5),np.uint8)
mask = cv2.erode(mask, kernel, iterations=2)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)

# similarly, dilations can increase the size of the ground objects
mask = thresh.copy()
kernel = np.ones((5,5),np.uint8)
mask = cv2.dilate(mask, kernel, iterations=2)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
---------------------------------------------
opening/Closing #
##################
import cv2
import numpy as np

img1 = cv2.imread('images/opening.png',0)
img2 = cv2.imread('images/closing.png',0)

kernel = np.ones((5,5),np.uint8)

opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel) # OE
cv2.imshow("img1", img1)
cv2.imshow("Opening", opening)
cv2.waitKey(0)

closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)  # CD
cv2.imshow("img2", img2)
cv2.imshow("Closing", closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

############################################
使用 webcam/usb                            #
https://github.com/NVIDIA-AI-IOT/jetcam   #
###########################################
import cv2
# you ==> 0
# instructor ==>1
#  選擇第1隻攝影機 ，
capture = cv2.VideoCapture(0)
while True:
#    從攝影機擷取一張影像
    returnValue, frame = capture.read()
#    開新視窗顯示圖片
    cv2.imshow("this is me!", frame)
    # cv2.waitKey(1000)：在1000ms内根据键盘输入返回一个值
    # cv2.waitKey(1)：在1ms内根据键盘输入返回一个值(沒按鍵盤傳回值 -1)
    # cv2.waitKey(1) & 0xFF = cv2.waitKey(1) & 11111111 = cv2.waitKey備案下的鍵值
    if cv2.waitKey(1) & 0xFF == ord('q'):    # 0xFF(十六進制) =255(十進制) =11111111(二進制)
        break
    pass
capture.release() # 釋放攝影機

color change to 黑白影像~: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)~~~~~~~~~~
import cv2
# you ==> 0
# instructor ==>1
capture = cv2.VideoCapture(10)
while True:
    returnValue, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("this is me!", gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    pass
capture.release()
cv2.destroyAllWindows()

demo10 程式不會停: 按下 "Stop"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import matplotlib.pyplot as plt

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)

k = 0
while k != 27:
    cv2.imshow("main window", originalImage)
    cv2.putText(originalImage, "left click, and drag", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    k = cv2.waitKey(20)
cv2.destroyAllWindows()

#######################################################################
回應函式 : OpenCV 有回應事件的 callback function: like mouse , keyboard  #
# cv2.setMouseCallback("main window", drawCircle)                    #
#######################################################################

import cv2
import matplotlib.pyplot as plt

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)


def drawCircle(action, x, y, flags, userdata):
    if action == cv2.EVENT_LBUTTONDOWN:  # mouse left down
        print("left bottom down, at({}.{})".format(x, y))
    elif action == cv2.EVENT_LBUTTONUP:  # mouse left up
        print("left bottom release up, at({}.{})".format(x, y))


cv2.namedWindow("main window")
cv2.setMouseCallback("main window", drawCircle)  # 透過action 傳callback
k = 0
while k != 27:  # 27 = esc
    cv2.imshow("main window", originalImage)
    cv2.putText(originalImage, "left click, and drag", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    k = cv2.waitKey(20)
cv2.destroyAllWindows()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import matplotlib.pyplot as plt

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)

center = [0, 0]


def drawCircle(action, x, y, flags, userdata):
    global center
    if action == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        print("left bottom down, at({}.{})".format(x, y))
        cv2.circle(originalImage, center, 1, (255, 255, 0), 6, cv2.LINE_AA)
    elif action == cv2.EVENT_LBUTTONUP:
        print("left bottom release up, at({}.{})".format(x, y))


WINDOW_NAME = "main window"
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, drawCircle)  # parameter :(windowName, function) 傳遞Function
k = 0
while k != 27:
    cv2.imshow(WINDOW_NAME, originalImage)
    cv2.putText(originalImage, "left click, and drag", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    k = cv2.waitKey(20)    # k is 鍵盤值 : 27 = esc
windows = cv2.destroyAllWindows()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import matplotlib.pyplot as plt

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)
imageCopy = originalImage.copy()
center = (0, 0)


def drawCircle(action, x, y, flags, userdata):
    global center
    if action == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        print("left bottom down, at({}.{})".format(x, y))
        cv2.circle(originalImage, center, 1, (255, 255, 0), 6, cv2.LINE_AA)
    elif action == cv2.EVENT_LBUTTONUP:
        print("left bottom release up, at({}.{})".format(x, y))


WINDOW_NAME = "main window"
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, drawCircle)
k = 0
while k != 27:
    cv2.imshow(WINDOW_NAME, originalImage)
    cv2.putText(originalImage, "left click, and drag", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    k = cv2.waitKey(20)
    if k == ord('c'):   # key =c ， 化的點去除
        originalImage = imageCopy.copy()
cv2.destroyAllWindows()
--------------------------------------------------------------------
import cv2
import matplotlib.pyplot as plt
import math

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)
imageCopy = originalImage.copy()
center = (0, 0)
bbox = (0, 0)

def drawCircle(action, x, y, flags, userdata):
    global center
    if action == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        print("left bottom down, at({}.{})".format(x, y))
        cv2.circle(originalImage, center, 1, (255, 255, 0), 6, cv2.LINE_AA)
    elif action == cv2.EVENT_LBUTTONUP:
        bbox = (x, y)
        radius = math.sqrt(math.pow(center[0] - bbox[0], 2) +
                           math.pow(center[1] - bbox[1], 2))
        cv2.circle(originalImage, center, int(radius), (0, 255, 255), 2, cv2.LINE_AA)
        print("left bottom release up, at({}.{})".format(x, y))

WINDOW_NAME = "main window"
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, drawCircle)
k = 0
while k != 27:
    cv2.imshow(WINDOW_NAME, originalImage)
    cv2.putText(originalImage, "left click, and drag", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    k = cv2.waitKey(20)
    if k == ord('c'):
        originalImage = imageCopy.copy()
cv2.destroyAllWindows()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import matplotlib.pyplot as plt
import math

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)
imageCopy = originalImage.copy()
#center = (0, 0)
#bbox = (0, 0)

def drawCircle(action, x, y, flags, userdata):
    global center
    if action == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        print("left bottom down, at({}.{})".format(x, y))
        cv2.circle(originalImage, center, 1, (255, 255, 0), 6, cv2.LINE_AA)
    elif action == cv2.EVENT_LBUTTONUP:
        bbox = (x, y)
        radius = math.sqrt(math.pow(center[0] - bbox[0], 2) +
                           math.pow(center[1] - bbox[1], 2))
        cv2.circle(originalImage, center, int(radius), (0, 255, 255), 2, cv2.LINE_AA)
        print("left bottom release up, at({}.{})".format(x, y))

WINDOW_NAME = "main window"
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, drawCircle)
k = 0
while k != 27:
    cv2.imshow(WINDOW_NAME, originalImage)
    cv2.putText(originalImage, "left click, and drag", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    k = cv2.waitKey(20)
    if k == ord('c'):
        originalImage = imageCopy.copy()
cv2.destroyAllWindows()
git config --list
more .gitconfig
pip install pillow
https://www.google.com/get/noto/#sans-hant

#########################################
#demo10 drqw 中文字                     ##
# 1 ####################################
import cv2
import matplotlib.pyplot as plt
import math
from PIL import ImageFont, ImageDraw, Image
import numpy as np

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)
imageCopy = originalImage.copy()
# center = (0, 0)
# bbox = (0, 0)

def drawCircle(action, x, y, flags, userdata):
    global center
    if action == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        print("left bottom down, at({}.{})".format(x, y))
        cv2.circle(originalImage, center, 1, (255, 255, 0), 6, cv2.LINE_AA)
    elif action == cv2.EVENT_LBUTTONUP:
        bbox = (x, y)
        radius = math.sqrt(math.pow(center[0] - bbox[0], 2) +
                           math.pow(center[1] - bbox[1], 2))
        cv2.circle(originalImage, center, int(radius), (0, 255, 255), 2, cv2.LINE_AA)
        print("left bottom release up, at({}.{})".format(x, y))


WINDOW_NAME = "main window"
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, drawCircle)
## 下載google 的Noto 字型(繁體中文)
FONT_PATH = 'C:\windows\fonts\mingliu.ttc'
## 載入中文字型
font = ImageFont.truetype(FONT_PATH, 24,encoding='utf-8')
# originalImage 轉成array
img_pil = Image.fromarray(originalImage)
# 轉換成draw 物件 for 加入中文字 -------------------------
draw = ImageDraw.Draw(img_pil)
k = 0
while k != 27:
    cv2.imshow(WINDOW_NAME, originalImage)
    draw.text((10, 200), "按下左鍵left click, and drag", (255, 0, 0),font=font)#fill=(255, 255, 0, 255))
    originalImage = np.array(img_pil)
    # cv2.putText(originalImage, "按下左鍵left click, and drag", (50, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    k = cv2.waitKey(20)
    if k == ord('c'):
        originalImage = imageCopy.copy()
cv2.destroyAllWindows()
2##################################################
# [fix1]  //page 65
import cv2
import matplotlib.pyplot as plt
import math
from PIL import ImageFont, ImageDraw, Image
import numpy as np

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)
imageCopy = originalImage.copy()


# center = (0, 0)
# bbox = (0, 0)


def drawCircle(action, x, y, flags, userdata):
    global center
    if action == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        print("left bottom down, at({}.{})".format(x, y))
        cv2.circle(originalImage, center, 1, (255, 255, 0), 6, cv2.LINE_AA)
    elif action == cv2.EVENT_LBUTTONUP:
        bbox = (x, y)
        radius = math.sqrt(math.pow(center[0] - bbox[0], 2) +
                           math.pow(center[1] - bbox[1], 2))
        cv2.circle(originalImage, center, int(radius), (0, 255, 255), 2, cv2.LINE_AA)
        print("left bottom release up, at({}.{})".format(x, y))


WINDOW_NAME = "main window"
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, drawCircle)
FONT_PATH = 'fonts/NotoSerifCJK-Regular.ttc'
font = ImageFont.truetype(FONT_PATH, 36)
img_pil = Image.fromarray(originalImage)
draw = ImageDraw.Draw(img_pil)
k = 0
while k != 27:
    cv2.imshow(WINDOW_NAME, originalImage)
    draw.text((10, 200), "按下左鍵left click, and drag", font=font, fill=(255, 255, 0, 255))
    originalImage = np.array(img_pil)
    # cv2.putText(originalImage, "按下左鍵left click, and drag", (50, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    k = cv2.waitKey(20)
    if k == ord('c'):
        originalImage = imageCopy.copy()
cv2.destroyAllWindows()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3 正確的 : page 63 相關 for draw 的 code 放進迴圈中##################################
import cv2
import matplotlib.pyplot as plt
import math
from PIL import ImageFont, ImageDraw, Image
import numpy as np

IMAGE1 = 'images/bg1.jpg'
originalImage = cv2.imread(IMAGE1, -1)
imageCopy = originalImage.copy()
# center = (0, 0)
# bbox = (0, 0)

def drawCircle(action, x, y, flags, userdata):
    global center
    if action == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        print("left bottom down, at({}.{})".format(x, y))
        cv2.circle(originalImage, center, 1, (255, 255, 0), 6, cv2.LINE_AA)
    elif action == cv2.EVENT_LBUTTONUP:
        bbox = (x, y)
        radius = math.sqrt(math.pow(center[0] - bbox[0], 2) +
                           math.pow(center[1] - bbox[1], 2))
        cv2.circle(originalImage, center, int(radius), (0, 255, 255), 2, cv2.LINE_AA)
        print("left bottom release up, at({}.{})".format(x, y))

WINDOW_NAME = "main window"
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, drawCircle)   # mouse Callback
FONT_PATH = 'fonts/NotoSerifCJK-Regular.ttc'
font = ImageFont.truetype(FONT_PATH, 36)
k = 0
while k != 27:
    cv2.imshow(WINDOW_NAME, originalImage)
    img_pil = Image.fromarray(originalImage)
    draw = ImageDraw.Draw(img_pil)
    draw.text((10, 200), "按下左鍵left click, and drag", font=font, fill=(255, 255, 0, 255))
    originalImage = np.array(img_pil)
    # cv2.putText(originalImage, "按下左鍵left click, and drag", (50, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    k = cv2.waitKey(20)
    if k == ord('c'):
        originalImage = imageCopy.copy()
cv2.destroyAllWindows()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pip install flask imutils
make a directory demo11
make a sub directory templates
demo11.py 放在demo11 目錄中
demo11.py ----------------------------------------------

from flask import Flask, render_template, Response
app = Flask(__name__)
if __name__ == '__main__':
     app.run(debug=True, threaded=True, use_reloader=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
index.html

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AIOCV Lab11</title>
</head>
<body>
<H1>Open CV integrate with Flask</H1>
<img src="{{url_for('video_feed')}}"/>
</body>
</html>

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
demo11.py

from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)
############~~~~~~~~~~~~~~~~~~~~~~~~~~~~
demo11.py ##  temtemp
#############
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            flag, encodedImage = cv2.imencode("*.jpg", outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage)
               + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
     app.run(debug=True, threaded=True, use_reloader=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from flask import Flask, render_template, Response
import cv2
import threading
from imutils.video import VideoStream
import imutils
import datetime

app = Flask(__name__)

outputFrame = None
lock = threading.Lock()
# default == 0
# instructor ==1
vs = VideoStream(src=0).start()


def getScreen(frameCount):
    global vs, outputFrame, lock
    total = 0
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        gray = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = datetime.datetime.now()
        cv2.putText(gray, timestamp.strftime("%A %d %B %Y %I:%M:%S:%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 0, 255), 1)
        total += 1
        with lock:
            outputFrame = gray.copy()


@app.route("/")
def index():
    return render_template("index.html")


def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            flag, encodedImage = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage)
               + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    t = threading.Thread(target=getScreen, args=(32,))
    t.daemon = True
    t.start()
    app.run(debug=True, threaded=True, use_reloader=False)
vs.stop()

#########################################################
cv2.imdecode()函数从指定的内存缓存中读取数据，并把数据转换(解码)成图像格式;主要用于从网络传输数据中恢复出图像。
cv2.imencode()函数是将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输。
Demo1 ======================================
import numpy as np
import urllib.request
import cv2
#
# url = 'http://www.pyimagesearch.com/wp-content/uploads/2015/01/google_logo.png'
# resp = urllib.request.urlopen(url)
# image = np.asarray(bytearray(resp.read()))
# image = cv2.imdecode(image, cv2.IMREAD_COLOR)
# cv2.imshow('URLImage',image)
# cv2.waitKey()
#
IMAGE_PATH1 = 'images/bg1.jpg'
# image1 = cv2.imread(IMAGE_PATH1, -1)
#
img = cv2.imread(IMAGE_PATH1, -1)
print(img)
print(img.shape) #
cv2.imshow('Load image', img)
cv2.waitKey()

img_encode = cv2.imencode('.png', img)[1]
data_encode = np.array(img_encode)
str_encode = data_encode.tobytes()
print("===str_encode===========================")
print(str_encode)
with open('img_encode.txt','wb') as f:
    f.write(str_encode)
    f.flush

Demo2=========================================
import numpy as np
import urllib.request
import cv2
IMAGE_PATH1 = 'images/bg1.jpg'
image1 = cv2.imread(IMAGE_PATH1)
img_encode = cv2.imencode('.jpg', image1)[1]
data_encode = np.array(img_encode)
str_encode = data_encode.tostring()
print(str_encode)
with open('img_encode1.txt','w') as f:
    f.write(str_encode)
    f.flush

with open('img_encode1.txt','r') as f:
    str_encode = f.read()

nparr = np.fromstring(str_encode, np.uint8)
img_decode = cv2.imdecode(nparr , cv2.IMREAD_COLOR)
cv2.imshow("img_decode",img_decode)
cv2.waitKey()