# demo 17 barcode
# cv2 QR code (实际用还是用ZBar(https://zbar.sourceforge.net/)比较好，ZBar还支持条形码的识别，
# OpenCV只有二维码的识别。)
# import cv2
# import os
# import matplotlib.pyplot as plt
#
# print(os.getcwd())
# BARCODE_FILE1 = 'images/barcode_sample.jpg'
# originalImage = cv2.imread(BARCODE_FILE1)
# data, bbox, extractedImage = cv2.QRCodeDetector().detectAndDecode(originalImage)
# print(type(data))
# print(type(bbox), bbox.shape)
# print(type(extractedImage))
# print(f"bar code content={data}, bbox={bbox}")
# print('show orgImage')
# cv2.imshow("originalimage", originalImage)
# ret_value = cv2.waitKey(0)
# if ret_value == ord('q'):
#     cv2.destroyWindow("originalimage")
#
# print('show extractedImage')
# cv2.imshow("extractedImage", extractedImage)
# ret_value = cv2.waitKey(0)
# if ret_value == ord('q'):
#     cv2.destroyWindow("extractedImage")
#
# OUTPUT_FILE = 'images/barcode_annotated.jpg'
# def drawBBox(originalImage, bbox, barcodeData):
#     COLOR1 = (255,0,255)
#     p1 = (bbox[0][0][0], bbox[0][0][1])
#     p2 = (bbox[0][1][0], bbox[0][1][1])
#     p3 = (bbox[0][2][0], bbox[0][2][1])
#     p4 = (bbox[0][3][0], bbox[0][3][1])
#     cv2.line(originalImage, p1, p2, COLOR1, 2)
#     cv2.line(originalImage, p2, p3, COLOR1, 2)
#     cv2.line(originalImage, p3, p4, COLOR1, 2)
#     cv2.line(originalImage, p4, p1, COLOR1, 2)
#     cv2.imwrite(OUTPUT_FILE, originalImage)
#     plt.imshow(originalImage)
#     plt.title(f"barcode:{barcodeData} with bounding box")
#     plt.show()
#
#
# if data != None:
#     drawBBox(originalImage, bbox, data)
# else:
#     print("QR code not Detected")
#=====================================================================
# import cv2
# import os
# from matplotlib import pyplot
#
# print(os.getcwd())
# BARCODE_FILE1 = 'images/barcode_sample.jpg'
# originalImage = cv2.imread(BARCODE_FILE1)
# data, bbox, extractedImage = cv2.QRCodeDetector().detectAndDecode(originalImage)
# print(type(data))
# print(type(bbox), bbox.shape)
# print(type(extractedImage))
# print(f"bar code content={data}, bbox={bbox}")
# cv2.imshow("original image", originalImage)
# cv2.imshow("image extract from cv2", extractedImage)
#
# OUTPUT_FILE = 'images/barcode_annotated.jpg'
# def drawBBox(originalImage, bbox, barcodeData):
#     COLOR1 = (128, 128, 0)
#     p1 = (bbox[0][0][0], bbox[0][0][1])
#     p2 = (bbox[0][1][0], bbox[0][1][1])
#     p3 = (bbox[0][2][0], bbox[0][2][1])
#     p4 = (bbox[0][3][0], bbox[0][3][1])
#     cv2.line(originalImage, p1, p2, COLOR1, 2)
#     cv2.line(originalImage, p2, p3, COLOR1, 2)
#     cv2.line(originalImage, p3, p4, COLOR1, 2)
#     cv2.line(originalImage, p4, p1, COLOR1, 2)
#     cv2.imwrite(OUTPUT_FILE, originalImage)
#     pyplot.imshow(originalImage)
#     pyplot.title(f"barcode:{barcodeData} with bounding box")
#     pyplot.show()
#
#
# if data != None:
#     drawBBox(originalImage, bbox, data)
# else:
#     print("QR code not Detected")

import cv2
image = cv2.imread('./images/barcode_sample.jpg')

qrCodeDetector = cv2.QRCodeDetector()
decodedText, points, _ = qrCodeDetector.detectAndDecode(image)

if points is not None:
    print(decodedText)
else:
    print("QR code not detected")

nrOfPoints = len(points)
print('len(points)=',nrOfPoints)

for i in range(nrOfPoints):
    nextPointIndex = (i+1) % nrOfPoints
    cv2.line(image, tuple(points[i][0]), tuple(points[nextPointIndex][0]), (255,0,0), 5)

print(decodedText)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


