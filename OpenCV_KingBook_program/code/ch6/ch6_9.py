# ch6_9.py
import numpy as np

image = np.random.randint(0,200, size=[3,5], dtype=np.uint8)
print(f"image = \n{image}")
print(f"修改前image.item(1,3) = {image.item(1,3)}")
image.itemset((1,3), 255)           # 修訂內容為 255
print("-"*70)
print(f"修改後image =\n{image}")
print(f"修改後image.item(1,3) = {image.item(1,3)}")







 
