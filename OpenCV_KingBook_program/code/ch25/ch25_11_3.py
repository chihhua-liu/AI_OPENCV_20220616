# ch25_11_3.py
import numpy as np

data = np.arange(3)
print(f"data = \n {data}")
x = np.repeat(data, 3)[:,np.newaxis]
print(f"After repeat = \n{x}")











