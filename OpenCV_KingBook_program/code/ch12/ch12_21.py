# ch12_21.py
import cv2
import numpy as np

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
print(f"MORPH_RECT \n {kernel}")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
print(f"MORPH_ELLIPSE \n {kernel}")
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
print(f"MORPH_CROSS \n {kernel}")


