# ch25_10.py
import numpy as np

data = np.arange(16).reshape(2,2,2,2)
print(f"data = \n {data}")
print(f"data = \n {np.vsplit(data,2)}")











