# ch31_2.py
from PIL import Image
import pytesseract

config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
text = pytesseract.image_to_string(Image.open('atq9305.jpg'),
                                   config=config)
print(f"車號是 : {text}")


