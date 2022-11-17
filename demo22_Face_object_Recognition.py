# # 1. Face Reconnition :臉部辨識(辨識人臉，確定人名)
# # python demo22_Face_object_Recognition.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle
# # python demo22_Face_object_Recognition.py --image images/adrian.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle
# # import the necessary packages
# import numpy as np
# import argparse
# import imutils
# import pickle
# import cv2
# import os
#
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=False,default='./images/adrian.jpg',
#     help="path to input image")
#
# ap.add_argument("-p", "--prototxt", required=True,
#                  help="path to Tensorflow 'deploy' pbtxt file")
#
# # ap.add_argument("-d", "--detector", required=False,default='./face_detection_model',
# #     help="path to OpenCV's deep learning face detector")
# ap.add_argument("-m", "--model", required=True,
#                  help="path to Caffe pre-trained model")
#
# ap.add_argument("-e", "--embedding-model", required=False,default='openface_nn4.small2.v1.t7',
#     help="path to OpenCV's deep learning face embedding model")
#
# ap.add_argument("-r", "--recognizer", required=False,default='output/recognizer.pickle',
#     help="path to model trained to recognize faces")
#
# ap.add_argument("-l", "--le", required=False,default='output/le.pickle',
#     help="path to label encoder")
#
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#     help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())
#
# # load our serialized face detector from disk
# print("[INFO] loading face detector...")
#
# dirname, filename = os.path.split(os.path.abspath(__file__))
# prototxt = os.path.join(dirname, args["prototxt"])
# model = os.path.join(dirname, args["model"])
# # protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
# # modelPath = os.path.sep.join([args["detector"],
# #     "res10_300x300_ssd_iter_140000.caffemodel"])
# detector = cv2.dnn.readNetFromCaffe(prototxt, model)
#
# # load our serialized face embedding model from disk
# print("[INFO] loading face recognizer...")
# embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
#
# # load the actual face recognition model along with the label encoder
# recognizer = pickle.loads(open(args["recognizer"], "rb").read())
# le = pickle.loads(open(args["le"], "rb").read())
#
# # load the image, resize it to have a width of 600 pixels (while
# # maintaining the aspect ratio), and then grab the image dimensions
# image = cv2.imread(args["image"])
# image = imutils.resize(image, width=600)
# (h, w) = image.shape[:2]
#
# # construct a blob from the image
# imageBlob = cv2.dnn.blobFromImage(
#     cv2.resize(image, (300, 300)), 1.0, (300, 300),
#     (104.0, 177.0, 123.0), swapRB=False, crop=False)
#
# # apply OpenCV's deep learning-based face detector to localize
# # faces in the input image
# detector.setInput(imageBlob)
# detections = detector.forward()
#
# # loop over the detections
# for i in range(0, detections.shape[2]):
#     # extract the confidence (i.e., probability) associated with the
#     # prediction
#     confidence = detections[0, 0, i, 2]
#
#     # filter out weak detections
#     if confidence > args["confidence"]:
#         # compute the (x, y)-coordinates of the bounding box for the
#         # face
#         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#         (startX, startY, endX, endY) = box.astype("int")
#
#         # extract the face ROI
#         face = image[startY:endY, startX:endX]
#         (fH, fW) = face.shape[:2]
#
#         # ensure the face width and height are sufficiently large
#         if fW < 20 or fH < 20:
#             continue
#
#         # construct a blob for the face ROI, then pass the blob
#         # through our face embedding model to obtain the 128-d
#         # quantification of the face
#         faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
#             (0, 0, 0), swapRB=True, crop=False)
#         embedder.setInput(faceBlob)
#         vec = embedder.forward()
#
#         # perform classification to recognize the face
#         preds = recognizer.predict_proba(vec)[0]
#         j = np.argmax(preds)
#         proba = preds[j]
#         name = le.classes_[j]
#
#         # draw the bounding box of the face along with the associated
#         # probability
#         text = "{}: {:.2f}%".format(name, proba * 100)
#         y = startY - 10 if startY - 10 > 10 else startY + 10
#         cv2.rectangle(image, (startX, startY), (endX, endY),
#             (0, 0, 255), 2)
#         cv2.putText(image, text, (startX, y),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#
# # show the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)
#------------------------------------------------------------------------
# recognize_video.py
# USAGE
# python demo22_Face_object_Recognition.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle

# # import the necessary packages
# from imutils.video import VideoStream
# from imutils.video import FPS
# import numpy as np
# import argparse
# import imutils
# import pickle
# import time
# import cv2
# import os
#
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--detector", required=True,
#     help="path to OpenCV's deep learning face detector")
# ap.add_argument("-m", "--embedding-model", required=True,
#     help="path to OpenCV's deep learning face embedding model")
# ap.add_argument("-r", "--recognizer", required=True,
#     help="path to model trained to recognize faces")
# ap.add_argument("-l", "--le", required=True,
#     help="path to label encoder")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#     help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())
#
# # load our serialized face detector from disk
# print("[INFO] loading face detector...")
# protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
# modelPath = os.path.sep.join([args["detector"],
#     "res10_300x300_ssd_iter_140000.caffemodel"])
# detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#
# # load our serialized face embedding model from disk
# print("[INFO] loading face recognizer...")
# embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
#
# # load the actual face recognition model along with the label encoder
# recognizer = pickle.loads(open(args["recognizer"], "rb").read())
# le = pickle.loads(open(args["le"], "rb").read())
#
# # initialize the video stream, then allow the camera sensor to warm up
# print("[INFO] starting video stream...")
# #vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# time.sleep(2.0)
#
# # start the FPS throughput estimator
# fps = FPS().start()
#
# # loop over frames from the video file stream
# while True:
#     # grab the frame from the threaded video stream
#     frame = vs.read()
#
#     # resize the frame to have a width of 600 pixels (while
#     # maintaining the aspect ratio), and then grab the image
#     # dimensions
#     frame = imutils.resize(frame, width=600)
#     (h, w) = frame.shape[:2]
#
#     # construct a blob from the image
#     imageBlob = cv2.dnn.blobFromImage(
#         cv2.resize(frame, (300, 300)), 1.0, (300, 300),
#         (104.0, 177.0, 123.0), swapRB=False, crop=False)
#
#     # apply OpenCV's deep learning-based face detector to localize
#     # faces in the input image
#     detector.setInput(imageBlob)
#     detections = detector.forward()
#
#     # loop over the detections
#     for i in range(0, detections.shape[2]):
#         # extract the confidence (i.e., probability) associated with
#         # the prediction
#         confidence = detections[0, 0, i, 2]
#
#         # filter out weak detections
#         if confidence > args["confidence"]:
#             # compute the (x, y)-coordinates of the bounding box for
#             # the face
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#
#             # extract the face ROI
#             face = frame[startY:endY, startX:endX]
#             (fH, fW) = face.shape[:2]
#
#             # ensure the face width and height are sufficiently large
#             if fW < 20 or fH < 20:
#                 continue
#
#             # construct a blob for the face ROI, then pass the blob
#             # through our face embedding model to obtain the 128-d
#             # quantification of the face
#             faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
#                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
#             embedder.setInput(faceBlob)
#             vec = embedder.forward()
#
#             # perform classification to recognize the face
#             preds = recognizer.predict_proba(vec)[0]
#             j = np.argmax(preds)
#             proba = preds[j]
#             name = le.classes_[j]
#
#             # draw the bounding box of the face along with the
#             # associated probability
#             text = "{}: {:.2f}%".format(name, proba * 100)
#             y = startY - 10 if startY - 10 > 10 else startY + 10
#             cv2.rectangle(frame, (startX, startY), (endX, endY),
#                 (0, 0, 255), 2)
#             cv2.putText(frame, text, (startX, y),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#
#     # update the FPS counter
#     fps.update()
#
#     # show the output frame
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#
#     # if the `q` key was pressed, break from the loop
#     if key == ord("q"):
#         break
#
# # stop the timer and display FPS information
# fps.stop()
# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#
# # do a bit of cleanup
# cv2.destroyAllWindows()
# vs.stop()
#--------------------------------------------------------------------------------
# # train_model.py ---------------
# # USAGE
# # python demo22_Face_object_Recognition.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle
# # import the necessary packages
# from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
# import argparse
# import pickle
#
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-e", "--embeddings", required=True,
#     help="path to serialized db of facial embeddings")
# ap.add_argument("-r", "--recognizer", required=True,
#     help="path to output model trained to recognize faces")
# ap.add_argument("-l", "--le", required=True,
#     help="path to output label encoder")
# args = vars(ap.parse_args())
#
# # load the face embeddings
# print("[INFO] loading face embeddings...")
# data = pickle.loads(open(args["embeddings"], "rb").read())
#
# # encode the labels
# print("[INFO] encoding labels...")
# le = LabelEncoder()
# labels = le.fit_transform(data["names"])
#
# # train the model used to accept the 128-d embeddings of the face and
# # then produce the actual face recognition
# print("[INFO] training model...")
# #recognizer = SVC(C=1.0, kernel="linear", probability=True)
# recognizer = SVC(C=1.0, gamma='scale', probability=True)
# recognizer.fit(data["embeddings"], labels)
#
# # write the actual face recognition model to disk
# f = open(args["recognizer"], "wb")
# f.write(pickle.dumps(recognizer))
# f.close()
#
# # write the label encoder to disk
# f = open(args["le"], "wb")
# f.write(pickle.dumps(le))
# f.close()
# ----------------------------------------------------------------------------------
# extract_embedings.py ---------------------------
# USAGE
# python demo22_Face_object_Recognition.py --dataset dataset --embeddings output/embeddings.pickle \
#    --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

# import the necessary packages
# from imutils import paths
# import numpy as np
# import argparse
# import imutils
# import pickle
# import cv2
# import os
#
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--dataset", required=True,
#     help="path to input directory of faces + images")
# ap.add_argument("-e", "--embeddings", required=True,
#     help="path to output serialized db of facial embeddings")
# ap.add_argument("-d", "--detector", required=True,
#     help="path to OpenCV's deep learning face detector")
# ap.add_argument("-m", "--embedding-model", required=True,
#     help="path to OpenCV's deep learning face embedding model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#     help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())
#
# # load our serialized face detector from disk
# print("[INFO] loading face detector...")
# protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
# modelPath = os.path.sep.join([args["detector"],"res10_300x300_ssd_iter_140000_fp16.caffemodel"])
# detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#
# # load our serialized face embedding model from disk
# print("[INFO] loading face recognizer...")
# embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
#
# # grab the paths to the input images in our dataset
# print("[INFO] quantifying faces...")
# imagePaths = list(paths.list_images(args["dataset"]))
#
# # initialize our lists of extracted facial embeddings and
# # corresponding people names
# knownEmbeddings = []
# knownNames = []
#
# # initialize the total number of faces processed
# total = 0
#
# # loop over the image paths
# for (i, imagePath) in enumerate(imagePaths):
#     # extract the person name from the image path
#     print("[INFO] processing image {}/{}".format(i + 1,
#         len(imagePaths)))
#     name = imagePath.split(os.path.sep)[-2]
#
#     # load the image, resize it to have a width of 600 pixels (while
#     # maintaining the aspect ratio), and then grab the image
#     # dimensions
#     image = cv2.imread(imagePath)
#     image = imutils.resize(image, width=600)
#     (h, w) = image.shape[:2]
#
#     # construct a blob from the image
#     imageBlob = cv2.dnn.blobFromImage(
#         cv2.resize(image, (300, 300)), 1.0, (300, 300),
#         (104.0, 177.0, 123.0), swapRB=False, crop=False)
#
#     # apply OpenCV's deep learning-based face detector to localize
#     # faces in the input image
#     detector.setInput(imageBlob)
#     detections = detector.forward()
#
#     # ensure at least one face was found
#     if len(detections) > 0:
#         # we're making the assumption that each image has only ONE
#         # face, so find the bounding box with the largest probability
#         i = np.argmax(detections[0, 0, :, 2])
#         confidence = detections[0, 0, i, 2]
#
#         # ensure that the detection with the largest probability also
#         # means our minimum probability test (thus helping filter out
#         # weak detections)
#         if confidence > args["confidence"]:
#             # compute the (x, y)-coordinates of the bounding box for
#             # the face
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#
#             # extract the face ROI and grab the ROI dimensions
#             face = image[startY:endY, startX:endX]
#             (fH, fW) = face.shape[:2]
#
#             # ensure the face width and height are sufficiently large
#             if fW < 20 or fH < 20:
#                 continue
#
#             # construct a blob for the face ROI, then pass the blob
#             # through our face embedding model to obtain the 128-d
#             # quantification of the face
#             faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
#                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
#             embedder.setInput(faceBlob)
#             vec = embedder.forward()
#
#             # add the name of the person + corresponding face
#             # embedding to their respective lists
#             knownNames.append(name)
#             knownEmbeddings.append(vec.flatten())
#             total += 1
#
# # dump the facial embeddings + names to disk
# print("[INFO] serializing {} encodings...".format(total))
# data = {"embeddings": knownEmbeddings, "names": knownNames}
# f = open(args["embeddings"], "wb")
# f.write(pickle.dumps(data))
# f.close()
#-----------------------------------------------------------------------------------
### iii_build_face_dataset.py  for save camera image  -------------------------------------------
# USAGE
# python demo22_Face_object_Recognition.py --cascade C:/Users/mikal/anaconda3/envs/openvino_env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml  --output dataset/yourname --start 0
# import the necessary packages
# from imutils.video import VideoStream
# import argparse
# import imutils
# import time
# import cv2
# import os
# import sys
#
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--cascade", required=True,
#                 help="path to where the face cascade resides")
#
# ap.add_argument("-o", "--output", required=True,
#                 help="path to output directory")
#
# ap.add_argument("-s", "--start", type=int, default=0,
#                 help="start number, default is 0")
# args = vars(ap.parse_args())
#
# # check if output directory is exist
# if not os.path.isdir(args['output']):
#     print(f'Output directory {args["output"]} does not exist, create first...')
#     sys.exit(2)
#
# # load OpenCV's Haar cascade for face detection from disk
# detector = cv2.CascadeClassifier(args["cascade"])
#
# # initialize the video stream, allow the camera sensor to warm up,
# # and initialize the total number of example faces written to disk
# # thus far
# print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# # vs = VideoStream(usePiCamera=True).start()
# time.sleep(2.0)
#
# idx = args['start']
# total = 0
#
# # loop over the frames from the video stream
# while True:
#     # grab the frame from the threaded video stream, clone it, (just
#     # in case we want to write it to disk), and then resize the frame
#     # so we can apply face detection faster
#     frame = vs.read()
#     orig = frame.copy()
#     frame = imutils.resize(frame, width=400)
#
#     # detect faces in the grayscale frame
#     rects = detector.detectMultiScale(
#         cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
#         minNeighbors=5, minSize=(30, 30))
#
#     # loop over the face detections and draw them on the frame
#     for (x, y, w, h) in rects:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # show the output frame
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#
#     # if the `p` key was pressed, write the *original* frame to disk
#     # so we can later process it and use it for face recognition
#     if key == ord("p"):
#         p = os.path.sep.join([args["output"], "{}.png".format(
#             str(idx).zfill(5))])
#         cv2.imwrite(p, orig)
#         idx += 1
#         total += 1
#         print(f'{total} pictures saved...')
#     # if the `q` key was pressed, break from the loop
#     elif key == ord("q"):
#         break
#
# # do a bit of cleanup
# print("[INFO] {} face images stored".format(total))
# print("[INFO] cleaning up...")
# cv2.destroyAllWindows()
# vs.stop()
