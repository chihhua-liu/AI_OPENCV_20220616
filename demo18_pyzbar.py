# pip install pyzbar
# demo18
from pyzbar import pyzbar
import numpy as np
import cv2
from pprint import pprint
FILENAME = 'images/sample.png'

def decode(image):
    decodedObjects = pyzbar.decode(image)
    print('type(decodedObjects) ================================')
    print(type(decodedObjects))
    for item in decodedObjects:
        # print(type(item))
        print('pprint(dir(item)) ================================')
        pprint(dir(item))
        print('tye={item.type},data={item.data} ================================')
        print(f'type={item.type},data={item.data}')
    return decodedObjects


def display(image, decoded):
    for d in decoded:
        points = d.polygon
        hull = points
        hull_length = len(hull)
        print(hull_length)
        for i in range(0, hull_length):
            cv2.line(image, hull[i], hull[(i + 1) % hull_length], (255, 0, 128), 3)
    cv2.imshow("result", image)
    cv2.waitKey(0)


image = cv2.imread(FILENAME)
decoded = decode(image)
display(image, decoded)
# book : Foundations of Machine Learning, Second Edition (https://cs.nyu.edu/~mohri/mlbook/)
