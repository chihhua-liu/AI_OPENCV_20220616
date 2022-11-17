# ch17_5.py
import numpy as np

data = np.array([[3, 9],
                 [8, 2],
                 [5, 3]]
                )
print(f"data = {data}")
max_i = data[:,0].argmax()
print(f"最大值索引 = {max_i}")
print(f"最大值     = {data[max_i][0]}")
print(f"對應值     = {data[max_i][1]}")
max_val = tuple(data[data[:,0].argmax()])
print(f"最大值配對 = {max_val}")













