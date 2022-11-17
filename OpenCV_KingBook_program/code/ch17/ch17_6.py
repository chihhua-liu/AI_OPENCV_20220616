# ch17_6.py
import numpy as np

data = np.array([[[186, 39]],
                 [[181, 44]],
                 [[180, 44]]]
                )
print(f"原始資料data = \n{data}")
n = len(data)
print("取3維內的陣列資料")
for i in range(n):                              # 列印 3 個座標點
    print(data[i])
print(f"資料維度   = {data.ndim}")              # 維度
max_i = data[:,:,0].argmax()                    # x 最大值索引索引
print(f"x 最大值索引 = {max_i}")                # 列印 x 最大值索引
right = tuple(data[data[:,:,0].argmax()][0])    # 最大值元組
print(f"最大值元組 = {right}")                  # 列印最大值元組       
min_i = data[:,:,0].argmin()                    # x 最小值索引索引
print(f"x 最小值索引 = {min_i}")                # 列印 x 最小值索引
left = tuple(data[data[:,:,0].argmin()][0])     # 最小值元組
print(f"最小值元組 = {left}")                   # 列印最小值元組











