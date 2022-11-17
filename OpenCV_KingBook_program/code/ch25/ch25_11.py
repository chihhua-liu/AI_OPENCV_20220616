# ch25_11.py
import numpy as np

data = np.arange(16).reshape(4,4)
print(f"data = \n {data}")
print(f"split = \n{np.hsplit(data,2)}")











