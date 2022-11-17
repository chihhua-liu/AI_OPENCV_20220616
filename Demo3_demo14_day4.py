# Advanced Process
# 補充 -------------------------------
# import sys
# if len(sys.argv) < 2:
#     print('no argument')
#     sys.exit()
# print('hello')
# print(sys.argv[0])  # is 程式name
# print(sys.argv[1])  # input command
# 1. video_image (use camera capture image) ===================================
# import numpy as np
# import cv2 as cv
# import sys
# if len(sys.argv) != 2:
#     print('Usage:',sys.argv[0],'<image name>')
#     sys.exit(2)
# # Capture video cam
# cap = cv.VideoCapture(0)
# while True:
#     ret, frame = cap.read()  # Capture frame-by-frame
#     cv.imshow('frame',frame) # Display the resulting frame
#     key = cv.waitKey(1)
#     if key & 0xFF == ord('q'):
#         break
#     if key & 0xFF == ord('p'):
#         cv.imwrite(sys.argv[1],frame)
#         break
# cap.release() # When everything done, release the capture
# cv.destroyAllWindows()
# # in CMD : command : demo3_demo14_day4.py test.jpg
# 2. play video (讀取影片) ============================================
# import numpy as np
# import cv2 as cv
# cap = cv.VideoCapture('images/chaplin.mp4')
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cv.imshow('frame',frame)
#     cv.waitKey(2)
# cap.release()
# cv.destroyAllWindows()
# 3. 儲存 影片 =========================================================
# import numpy as np
# import cv2
# import sys
#
# if len(sys.argv) != 2:
#     print('Usage:', sys.argv[0], '<video name>')
#     sys.exit(2)
#
# cap = cv2.VideoCapture(0)
#
# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(sys.argv[1], fourcc, 20.0, (640, 480))
#
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         # write the frame
#         out.write(frame)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()
# # in CMD : command : demo3_demo14_day4.py output.avi
# 4. fix 取像大小:640* 480 儲存 影片 ========================================
# import numpy as np ; import cv2 ; import sys
# if len(sys.argv) != 2:
#     print('Usage:', sys.argv[0], '<video name>')
#     sys.exit(2)
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # fix 取像大小:640* 480
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(sys.argv[1], fourcc, 20.0, (640, 480))
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         out.write(frame)  # write the frame
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# cap.release()  # Release everything if job is finished
# out.release() ; cv2.destroyAllWindows()
# 補充  imutils  主要是要用來進行影像處理 像是平移, 旋轉, 縮放, 顯示, 骨架化=====================================================
# import imutils
# import cv2 as cv2
# import matplotlib.pyplot as plt
# import numpy as np
# FILE_NAME = 'images/lena_color.jpg'
# img = cv2.imread(FILE_NAME,1)
# cv2.imshow('org img', img)
# print(img.shape)
# cv2.waitKey(0)
# translated = imutils.translate(img,50,50)  #1. 平移
# cv2.imshow('t-image', translated)
# cv2.waitKey(0)
#
# resized = imutils.resize(img,width=200)  # 2.縮放: 指定寬度，會自動計算相應比例高度，還有引數height
# cv2.imshow('re-image', resized)
# cv2.waitKey(0)
#
# rotated = imutils.rotate(img, 90)   # 3.旋轉: 逆時針旋轉
# cv2.imshow('r0-image', rotated)
# cv2.waitKey(0)
# # rotated_round = imutils.rotate_bound(image, 90) # 順時針旋轉
# # 4 骨架提取（邊緣提取） bug
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # skeleton = imutils.skeletonize(gray, size=(3, 3))  # size=(7, 7)) is (structuring element)
# # cv2.imshow('sk-image', skeleton)
# # cv2.waitKey(0)
# #cv2.destoryAllWindows()
# #cv2.destroyAllWindows()
# # 5.使用Matplotlib展示图片
# plt.imshow(imutils.opencv2matplotlib(img))
# plt.show()
# # 6  自动边缘检测Automatic Canny Edge Detection
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edgeMap = imutils.auto_canny(gray)
# print(edgeMap.shape)
# cv2.imshow('AutoEdge', edgeMap)
# cv2.waitKey(0)
# #源码
# def auto_canny(image, sigma=0.33):
#     # compute the median of the single channel pixel intensities
#     v = np.median(image)
#     # apply automatic Canny edge detection using the computed median
#     lower = int(max(0, (1.0 - sigma) * v))
#     upper = int(min(255, (1.0 + sigma) * v))
#     edged = cv2.Canny(image, lower, upper)
#     # return the edged image
#     return edged
# 5.video image both ==bug==============================================
# import numpy as np
# import imutils
# from imutils.video import VideoStream
# from imutils.video import FPS
# import cv2 as cv2
# import sys
# import time
# if len(sys.argv) != 2:
#     print('Usage:', sys.argv[0], '<image name>')
#     sys.exit(2)
# # initialize the video stream, allow the cammera sensor to warmup,
# # and initialize the FPS counter
# print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# # vs = VideoStream(usePiCamera=True).start()
# time.sleep(2.0)
# fps = FPS().start()
# while True:
#     # grab the frame from the threaded video stream
#     frame = vs.read()
#     # Display the resulting frame
#     cv2.imshow('frame', frame)
#     key = cv2.waitKey(1)
#     fps.update()
#     if key & 0xFF == ord('q'):
#         break
#     if key & 0xFF == ord('p'):
#         cv2.imwrite(sys.argv[1], frame)
#         break
# fps.stop()
# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# # do a bit of cleanup
# vs.stop()
# # When everything done, release the capture
# cv.destroyAllWindows()
# feature sift 找關鍵點 =======================================================
# import numpy as np
# import argparse
# import cv2
# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required = True,
#     help = 'Path to the image')
# args = vars(ap.parse_args())
# desfile = args['image'].rsplit('.',maxsplit=1)[0]+'.npy'
# img = cv2.imread(args['image'])
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT_create()# cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)
# # directly find keypoints and descriptors in a single step
# kp, des = sift.detectAndCompute(gray,None)
# img=cv2.drawKeypoints(gray,kp,img)
# np.save(desfile, des)
# cv2.imshow("SIFT", img)
# cv2.waitKey(0)
# cv.imwrite('sift_keypoints.jpg',img)
#===========================================
# import numpy as np
# import cv2
# def SIFT_feature_detection( f ):
# 	g = cv2.cvtColor( f, cv2.COLOR_GRAY2BGR )
# 	sift = cv2.SIFT_create()
# 	kp = sift.detect( f, None )
# 	g = cv2.drawKeypoints( f, kp, g )
# 	return g
# def main( ):
# 	img1 = cv2.imread( "images/Blox.bmp", 0 )
# 	img2 = SIFT_feature_detection( img1 )
# 	cv2.imshow( "Original Image", img1 )
# 	cv2.imshow( "SIFT Features", img2 )
# 	cv2.waitKey( 0 )
# main( )
# SURF in OpenCV -因為專利 新版沒有了------------------------------------------------------
# import numpy as np
# import cv2
#
# def SURF_feature_detection( f ):
# 	g = cv2.cvtColor( f, cv2.COLOR_GRAY2BGR )
# 	surf =cv2.SURF_create()    #cv2.xfeatures2d.SURF_create()
# 	kp = surf.detect( f, None )
# 	g = cv2.drawKeypoints( f, kp, g, flags = 4 )
# 	return g
# def main( ):
# 	img1 = cv2.imread( "images/Blox.bmp", 0 )
# 	img2 = SURF_feature_detection( img1 )
# 	cv2.imshow( "Original Image", img1 )
# 	cv2.imshow( " SURF Features", img2 )
# 	cv2.waitKey( 0 )
# main( )
# FLANN (Fast Library for Approximate Nearest Neighbors)=====================
# 快速近似最近鄰居搜索程式庫 : 依據關鍵點找匹配
# import numpy as np ;
# import argparse
# import cv2
# from matplotlib import pyplot as plt
#
# FILE_NAME = 'images/aiotbooks.jpg'
# gray = cv2.imread(FILE_NAME,0)
# cv2.imshow('aiotbooks',gray)
# cv2.waitKey(0)
# FILE_NAME1 = 'images/aiotimage.jpg'
# desimg = cv2.imread(FILE_NAME1,0)
# cv2.imshow('aiotimage',desimg)
# cv2.waitKey(0)
#
# sift = cv2.SIFT_create()
# kp, des = sift.detectAndCompute(gray,None)
# print('kp=',kp)
# kp2, des2 = sift.detectAndCompute(desimg,None)
# print('kp2=',kp2)
# # # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
# flann = cv2.FlannBasedMatcher(index_params,search_params)
# des1 = np.float32(des)
# des2 = np.float32(des2)
# matches = flann.knnMatch(des,des2,k=2)
# matchesMask = [[0,0] for i in range(len(matches))]
# # # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         matchesMask[i]=[1,0]
#         print(m)
#
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = 0)
# img3 = cv2.drawMatchesKnn(gray,kp,desimg,kp2,matches,None,**draw_params)
# plt.imshow(img3,)
# plt.show()
# flann _ sift match2 =bug ==================================================
# import numpy as np
# import argparse
# import cv2
#
# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required = True,
#     help = 'Path to the image')
# ap.add_argument('-d', '--descriptor', required = True,
#     help = 'Path to feature descriptor')
#
# args = vars(ap.parse_args())
#
# desfile = args['descriptor']
# desdata = np.load(desfile)
# img = cv2.imread(args['image'])
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# sift = cv2.xfeatures2d.SIFT_create()
# # kp = sift.detect(gray,None)
# # directly find keypoints and descriptors in a single step
# kp, des = sift.detectAndCompute(gray,None)
#
# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
#
# flann = cv2.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des,desdata,k=2)
#
# # minimum number of matches
# MIN_MATCH_COUNT = 30
# good = []
#
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         good.append(kp[m.queryIdx])
#
# if len(good) > MIN_MATCH_COUNT:
#     print('{} is a match! ({})'.format(desfile,len(good)))
# else:
#     print(desfile,'is not a match')
#
# img2=cv2.drawKeypoints(img,good,img)
# cv2.imshow("SIFT", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #=====================================================================
# in CMD : ./Python-sys-argv.py 123
# 結果 :
# hello
# ./Python-sys-argv.py
# 123
# import cv2
# # for first camera, select 0
# capture = cv2.VideoCapture(0)
# counter = 0
# IMAGE_NAME = 'images/%d.jpg'
# while True:
#     ret, frame = capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     elif cv2.waitKey(1) & 0xFF == ord('s'):
#         cv2.imwrite(IMAGE_NAME % counter, frame)
#         counter += 1
# capture.release()
# cv2.destroyAllWindows()
#=========================================
# import cv2#
# # for first camera, select 0s
# capture = cv2.VideoCapture(0)
# counter = 0
# IMAGE_NAME = 'images/%d.jpg'#
# while True:
#     ret, frame = capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray)
#     intputKey = cv2.waitKey(1)
#     if intputKey & 0xFF == ord('q'):
#         break
#     elif intputKey & 0xFF == ord('s'):
#         filename = IMAGE_NAME % counter
#         cv2.imwrite(filename, gray)
#         print(f"file {filename} saved")
#         counter += 1
# capture.release()
# cv2.destroyAllWindows()
#----------------------------------------
# demo12_webcam_2_category.py (get image & write file) for demo 14
# import cv2
# # for first camera, select 0
# capture = cv2.VideoCapture(1)
# p_counter = 0
# n_counter = 0
# POSITIVE_NAME = 'images/positive/%d.jpg'
# NEGATIVE_NAME = 'images/negative/%d.jpg'
#
# while True:
#     ret, frame = capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray)
#     inputKey = cv2.waitKey(1)
#     if inputKey & 0xFF == ord('q'):
#         break
#     elif inputKey & 0xFF == ord('p'):
#         filename = POSITIVE_NAME % p_counter
#         cv2.imwrite(filename, gray)
#         print(f"positive file {filename} saved")
#         p_counter += 1
#     elif inputKey & 0xFF == ord('n'):
#         filename = NEGATIVE_NAME % n_counter
#         cv2.imwrite(filename, gray)
#         print(f"negative file {filename} saved")
#         n_counter += 1
# capture.release()
# cv2.destroyAllWindows()
#------------------------------------------------------
# # Demo 13  dataset
# import glob
# import os
# import subprocess
# import uuid
# import PIL.Image
# import cv2
# import torch.utils.data
#
#
# class ImageClassificationDataSet(torch.utils.data.Dataset):
#     def __init__(self, directory, categories, transform=None):
#         self.categories = categories
#         self.directory = directory
#         self.transform = transform
#         self._refresh()
#
#     def __len__(self):
#         return len(self.annotations)
#
#     def __getitem__(self, index):
#         ann = self.annotations[index]
#         image = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)
#         image = PIL.Image.fromarray(image)
#         if self.transform is not None:
#             image = self.transform(image)
#         return image, ann['category_index']
#
#     def get_count(self, category):
#         i = 0
#         for a in self.annotations:
#             if a['category'] == category:
#                 i += 1
#         return i

#     def _refresh(self):
#         print("get actual data")
#         self.annotations = []
#         for category in self.categories:
#             category_index = self.categories.index(category)
#             for image_path in glob.glob(os.path.join(self.directory, category, '*.jpg')):
#                 self.annotations += [{
#                     'image_path': image_path,
#                     'category_index': category_index,
#                     'category': category
#                 }]
#==================================================================================
###############################################
# demo12" : get image from camera  for demo14  #
###############################################
# import cv2
# # for first camera, select 0
# capture = cv2.VideoCapture(0)
# p_counter = 0
# n_counter = 0
# POSITIVE_NAME = 'image/positive/%d.jpg'
# NEGATIVE_NAME = 'image/negative/%d.jpg'
#
# while True:
#     ret, frame = capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray)
#     inputKey = cv2.waitKey(1)
#     if inputKey & 0xFF == ord('q'):
#         break
#     elif inputKey & 0xFF == ord('p'):
#         filename = POSITIVE_NAME % p_counter
#         cv2.imwrite(filename, frame)
#         print(f"positive file {filename} saved")
#         p_counter += 1
#     elif inputKey & 0xFF == ord('n'):
#         filename = NEGATIVE_NAME % n_counter
#         cv2.imwrite(filename, frame)
#         print(f"negative file {filename} saved")
#         n_counter += 1
# capture.release()
# cv2.destroyAllWindows()

####################################################################
##### demo14 pytorch_retrain_model : train resnet18 Model (遷移學習) #=============
####################################################################
# https://pytorch.org/vision/stable/transforms.html
# import torchvision.transforms as transforms
# from demo13_dataset import ImageClassificationDataSet
# import torchvision
# import torch
# import torch.nn.functional as F
#
# TASK = 'image'
# CATEGORIES = ['positive', 'negative']
# DATASETS = ['image']
# # transforms.ColorJitter 改变图像的属性：
# # # 亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue)
# ## Normalize()函数的作用是将数据转换为标准高斯分布
# # 即逐个channel的对图像进行标准化（均值变为0 00，标准差为1 11），可以加快模型的收敛
# # mean：各通道的均值
# # std：各通道的标准差
#
# TRANSFORMS = transforms.Compose([
#     transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     # mean = [0.485, 0.456, 0.406] ,std = [0.229, 0.224, 0.225]
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
# datasets = {}
# for name in DATASETS:
#     datasets[name] = ImageClassificationDataSet(name, CATEGORIES, TRANSFORMS)
# dataset = datasets[DATASETS[0]]
# print('DATASETS[0] = ',DATASETS[0] )
# print(f"{TASK} task with {CATEGORIES} categories defined")
#
# dataset._refresh()
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
# device = torch.device('cpu')
# model = torchvision.models.resnet18(pretrained=True)
# model.fc = torch.nn.Linear(512, len(dataset.categories))
# model = model.to(device)
# model.train()
#
# print(model)
# epochs = 10
# optimizer = torch.optim.Adam(model.parameters())
# while epochs > 0:
#     i = 0
#     sum_loss = 0.0
#     error_count = 0.0
#     for images, labels in iter(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = F.cross_entropy(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         error_count += len(torch.nonzero(outputs.argmax(1) - labels, as_tuple=False).flatten())
#         count = len(labels.flatten())
#         i += count
#         sum_loss += float(loss)
#     print("[{}],loss={},accuracy={}".format(epochs, sum_loss / i, 1.0 - error_count / i))
#     epochs = epochs - 1
#
# MODEL_WEIGHT = 'model/weight_only'
# WHOLE_MODEL = 'model/model_and_weight'
# torch.save(model.state_dict(), MODEL_WEIGHT)
# torch.save(model,WHOLE_MODEL)

# #補充============================================
# import os
#
# class stree():
#     def tree_s():
#         ina = input("要顯示的目錄：")
#         for root, dirs, files in os.walk(ina):
#             print(root)
#             for name in files:
#                 print('files: os.path.join(root, name)',os.path.join(root, name))
#             for name in dirs:
#                 print('dirs os.path.join(root, name)',os.path.join(root, name))
#         return -1  # 遞迴EOF
#
#
# if __name__ == "__main__": #讓import不輸出
#     stree.tree_s()

# 如果要整理一堆在目錄裡之相同副檔名或類似名稱的檔案，就可使用glob。
# glob.glob("要顯示的路徑檔案或方法")
# from glob import glob
# fsA = glob("exist.py"), glob("*dir.py"), glob("*.*")
# for file in fsA:
#     print(file)

#####################################
#  demo15_load_model_live_preview ### ==============================================
#####################################
# # demo15_load_model_live_preview (verify image use camera image)
# import cv2
# import torchvision.transforms as transforms
# import torchvision
# import torch
# import PIL
#
# MODEL_WEIGHT = 'model/weight_only'
# # for first camera, use 0
# cap = cv2.VideoCapture(0)
#
# device = torch.device('cpu')
# model1 = torchvision.models.resnet18()  # CNN Model
# print('model1 = ',model1)
# model1 = model1.to(device)
# model1.fc = torch.nn.Linear(512, 2)  # 最後一個fc( input 512, 分2類)
# model1.load_state_dict(torch.load(MODEL_WEIGHT))
#
# TRANSFORMS = transforms.Compose([
#     transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),   # 亮度
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # mean & std
# ])
#
# def verify(f):
#     image = f.to(device)
#     image = torch.reshape(image, [1, 3, 224, 224])  # CNN for 4D
#     output = model1(image)
#     if output.argmax(1) == 1:
#         print('down')
#     else:
#         print("up")
#
# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray)
#     inputKey = cv2.waitKey(1) & 0xFF  # cv2.waitKey(1): 取按键的ASCII值，0xFF 返回值最后八位
#     if inputKey == ord('q'):
#         break
#     elif inputKey == ord('v'):
#         print("call pytorch")
#         frame = PIL.Image.fromarray(frame) # transfor tensor for pytorch
#         frame = TRANSFORMS(frame)
#         verify(frame)
#
# cap.release()
# cv2.destroyAllWindows()

# 補充 :Softmax : https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html====
#  torch.nn.Softmax(dim=None) : a Tensor of the same dimension and shape as the input with values in the range [0, 1]
# import torch.nn as nn
# import torch
# m = nn.Softmax(dim=1) # range [0, 1] 像機率分布 (加起來=1)
# input = torch.randn(2, 3)
# print("input=\n",input)
# output = m(input)
# print("output = ",output)
############################
# demo16_compare_model    ##===============================================================
############################
# import torch
# import torchvision
# from torchvision import transforms
# from demo13_dataset import ImageClassificationDataSet
#
# TRANSFORMS = transforms.Compose([
#     transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
# MODEL_WEIGHT = 'model/weight_only'
# WHOLE_MODEL = 'model/model_and_weight'
#
# device = torch.device('cpu')
# model1 = torchvision.models.resnet18()
# model1 = model1.to(device)
# model1.fc = torch.nn.Linear(512, 2)
# model1.load_state_dict(torch.load(MODEL_WEIGHT))
#
# model2 = torch.load(WHOLE_MODEL)
# print("--------------------------------------------")
# def compare_models(m1, m2):
#     models_differ = 0
#     for key_item1, key_item2 in zip(m1.state_dict().items(), m2.state_dict().items()):
#         if torch.equal(key_item1[1], key_item2[1]):
#             pass
#         else:
#             models_differ += 1
#             if key_item1[0] == key_item2[0]:
#                 print("mismatch found at:{}".format(key_item1[0]))
#             else:
#                 raise Exception
#     if models_differ == 0:
#         print("model match perfectly")
# compare_models(model1, model2)  # 'model/weight_only' &  'model/model_and_weight' 結果一樣

# import torch
# import torchvision
# from torchvision import transforms
# from demo13_dataset import ImageClassificationDataSet
#
# TRANSFORMS = transforms.Compose([
#     transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
# MODEL_WEIGHT = 'model/weight_only'
# WHOLE_MODEL = 'model/model_and_weight'
#
# device = torch.device('cpu')
# model1 = torchvision.models.resnet18()
# model1 = model1.to(device)
# model1.fc = torch.nn.Linear(512, 2)
# model1.load_state_dict(torch.load(MODEL_WEIGHT))
#
# model2 = torch.load(WHOLE_MODEL)
# print("---------------------------------------------")
# def compare_models(m1, m2):
#     models_differ = 0
#     print(m1.state_dict())
#     print(m2.state_dict())
#     for key_item1, key_item2 in zip(m1.state_dict().items(), m2.state_dict().items()):
#         if torch.equal(key_item1[1], key_item2[1]):
#             pass
#         else:
#             models_differ += 1
#             if key_item1[0] == key_item2[0]:
#                 print("mismatch found at:{}".format(key_item1[0]))
#             else:
#                 raise Exception
#     if models_differ == 0:
#         print("model match perfectly")
#
#
# compare_models(model1, model2)
# FILE1 = 'model/model1.txt'
# FILE2 = 'model/model2.txt'
# with open(FILE1, 'w') as file1:
#     file1.write(str(model1))
# with open(FILE2, 'w') as file2:
#     file2.write(str(model2))
