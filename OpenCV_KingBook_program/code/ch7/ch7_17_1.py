# ch7_17_1.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def cv2_Chinese_Text(img,text,left,top,textColor,fontSize):
    ''' 建立中文字輸出 '''
# 影像轉成 PIL影像格式
    if (isinstance(img,np.ndarray)):   
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)              # 建立PIL繪圖物件
    fontText = ImageFont.truetype(          # 建立字型 - 新細明體
                "C:\Windows\Fonts\mingliu.ttc",     # 新細明體
                fontSize,                   # 字型大小
                encoding="utf-8")           # 編碼方式
    draw.text((left,top),text,textColor,font=fontText)  # 繪製中文字
# 將PIL影像格式轉成OpenCV影像格式
    return cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

img = cv2.imread("antarctic.jpg")
img = cv2_Chinese_Text(img, "我在南極", 220, 100, (0,0,255), 50)

cv2.imshow("Antarctic",img)                 
cv2.waitKey(0)                                      
cv2.destroyAllWindows()                     # 刪除所有視窗









 
