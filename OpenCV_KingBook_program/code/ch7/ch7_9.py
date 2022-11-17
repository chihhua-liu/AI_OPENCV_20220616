# ch7_9.py
import numpy as np

print("回傳單3個元素的陣列, 值是0(含)至256(不含)的隨機數")
arr = np.random.randint(0,256, size=3)
print(type(arr))
print(arr)
print("將陣列改為串列")
print(arr.tolist())














 
