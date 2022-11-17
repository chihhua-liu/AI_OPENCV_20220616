# USAGE
# python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

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
#
# ap.add_argument("-e", "--embeddings", required=True,
#     help="path to output serialized db of facial embeddings")
#
# ap.add_argument("-d", "--detector", required=True,
#     help="path to OpenCV's deep learning face detector")
#
# ap.add_argument("-m", "--embedding-model", required=True,
#     help="path to OpenCV's deep learning face embedding model")
#
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
# print("=======================================================")
# print('imagePaths',imagePaths)
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
### iii_build_face_dataset.py : SAVE FACE IMAGE -------------------------------------------
# USAGE
# python extract_embeddings.py --cascade C:/Users/mikal/anaconda3/envs/openvino_env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml  --output dataset/yourname --start 0

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
import sys

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path to where the face cascade resides")

ap.add_argument("-o", "--output", required=True,
                help="path to output directory")

ap.add_argument("-s", "--start", type=int, default=0,
                help="start number, default is 0")
args = vars(ap.parse_args())

# check if output directory is exist
if not os.path.isdir(args['output']):
    print(f'Output directory {args["output"]} does not exist, create first...')
    sys.exit(2)

# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

idx = args['start']
total = 0

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, clone it, (just
    # in case we want to write it to disk), and then resize the frame
    # so we can apply face detection faster
    frame = vs.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=400)

    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30))

    # loop over the face detections and draw them on the frame
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `p` key was pressed, write the *original* frame to disk
    # so we can later process it and use it for face recognition
    if key == ord("p"):
        p = os.path.sep.join([args["output"], "{}.png".format(
            str(idx).zfill(5))])
        cv2.imwrite(p, orig)
        idx += 1
        total += 1
        print(f'{total} pictures saved...')
    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()


