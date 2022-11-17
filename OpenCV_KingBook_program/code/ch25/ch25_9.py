# ch25_9.py
import numpy as np

data = np.arange(8).reshape(2,2,2)
print(f"data = \n {data}")
print(f"split = \n{np.vsplit(data,2)}")











