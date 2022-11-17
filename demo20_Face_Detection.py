# Car_plate.py :EDU--------------
# import cv2
# import time
# # load test iamge
# test1 = cv2.imread('images/car_plate.jpg')
# # convert the test image to gray image as opencv face detector expects gray images
# gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
# # load cascade classifier training file for haarcascade
# # C:/Users/mikal/anaconda3/envs/openvino_env/Lib/site-packages/cv2/data/haarcascade_russian_plate_number.xml
# haar_face_cascade = cv2.CascadeClassifier('C:/Users/mikal/anaconda3/envs/openvino_env/Lib/site-packages/cv2/data/haarcascade_russian_plate_number.xml')
# # let's detect multiscale (some images may be closer to camera than others) images
# cars = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
# # print the number of car plates found
# print('Car plate found: ', len(cars))
# # go over list of car plates and draw them as rectangles on original colored
# for (x, y, w, h) in cars:
#     cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)
# cv2.imshow('Test Imag', test1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# fdetect1.py : for frontal face -----------------------------------------------------
# import cv2
# import time
# # load test iamge
# test1 = cv2.imread('images/mybaby.jpg')
# # convert the test image to gray image as opencv face detector expects gray images
# gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray_img",gray_img)
# cv2.waitKey(0)
# # # load cascade classifier training file for haarcascade
# haar_face_cascade = cv2.CascadeClassifier('C:/Users/mikal/anaconda3/envs/openvino_env/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
# # # let's detect multiscale (some images may be closer to camera than others) images
# faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
# # # print the number of faces found
# print('Faces found: ', len(faces))
# print("faces=",faces)
# for (x, y, w, h) in faces:
#      cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)
# cv2.imshow('Test Imag', test1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# fdetect3 ------------------------------------------------------------------------------
# import cv2
# import time
# import sys
# import numpy as np
# # from picamera import PiCamera
# import imutils
# from imutils.video import VideoStream
#
# img_name = '/home/pi/image.jpg'
#
# # initialize the video stream, allow the cammera sensor to warmup,
# print("[INFO] starting video stream...")
# # vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# time.sleep(2.0)
#
# # wait for press key
# while True:
#     # grab the frame from the threaded video stream
#     frame = vs.read()
#
#     # Display the resulting frame
#     cv2.imshow('frame', frame)
#
#     key = cv2.waitKey(1)
#     if key & 0xFF == ord('q'):
#         img = None
#         break
#
#     if key & 0xFF == ord('p'):
#         img = frame
#         break
#
# # stop video stream
# vs.stop()
# cv2.destroyAllWindows()
#
# if img is None:  # press 'q' to quit
#     sys.exit(2)
#
# # convert the image to gray
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # load cascade classifier training file for haarcascade
# haar_face_cascade = cv2.CascadeClassifier('C:/Users/mikal/anaconda3/envs/openvino_env/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
#
# # let's detect multiscale (some images may be closer to camera than others) images
# faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);
#
# # print the number of faces found
# print('Faces found: ', len(faces))
#
# # list of faces and draw them as rectangles on original colored
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


