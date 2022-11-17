import numpy as np
# row1 = [1, 2, 3]
# arr1 = np.array(row1, ndmin=2)  # np.array(list)
# print(f"陣列維度 = {arr1.ndim}")
# print(f"陣列外型 = {arr1.shape}")
# print(f"陣列大小 = {arr1.size}")
# print("陣列內容")
# print(arr1)
# print("-"*70)
# row2 = [4, 5, 6]
# arr2 = np.array([row1,row2], ndmin=2)
# print(f"陣列維度 = {arr2.ndim}")
# print(f"陣列外型 = {arr2.shape}")
# print(f"陣列大小 = {arr2.size}")
# print("陣列內容")
# print(arr2)
# print("-"*70)
# x = np.array([[1, 2, 3], [4, 5, 6]])
# print(f"陣列維度 = {x.ndim}")
# print(f"陣列外型 = {x.shape}")
# print(f"陣列大小 = {x.size}")
# print("陣列內容")
# print(x)
# #-------------------------------------------------
# import numpy as np
# x = np.array([[1, 2, 3], [4, 5, 6]])
# print(x[0][2])
# print(x[1][2])
# # 或是
# print(x[0, 2])
# print(x[1, 2])
# #---------------------------------------------------
# import numpy as np
# import cv2
# x1 = np.zeros(3)
# print(x1)
# print("-"*70)
# x2 = np.zeros((2, 3), dtype=np.uint8)   # for 初始化
# print(x2)
# print("-"*70)
# fig = np.zeros((50, 200), dtype=np.uint8)
# print(fig)
# cv2.imshow("fig", fig)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#--------------------------------------------
# import numpy as np
# import cv2
# x1 = np.ones(3)
# print(x1)
# print("-"*70)
# x2 = np.ones((2, 3), dtype=np.uint8)
# print(x2)
# print("-"*70)
# fig = np.ones((50, 200), dtype=np.uint8) * 255
# print(fig)
# cv2.imshow("fig", fig)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #----------------------------------------------------
# x1 = np.empty(3)
# print(x1)
# print("-"*70)
# x2 = np.empty((2, 3), dtype=np.uint8)
# print(x2)
#--------------------------------------------------
# x1 = np.random.randint(10, 20)
# print("回傳值是10(含)至20(不含)的單一隨機數")
# print(x1)
# print("-"*70)
# print("回傳一維陣列10個元素, 值是1(含)至5(不含)的隨機數")
# x2 = np.random.randint(1, 5, 10)
# print(x2)
# print("-"*70)
# print("回傳單3*5陣列, 值是0(含)至10(不含)的隨機數")
# x3 = np.random.randint(10, size=(3, 5))
# print(x3)
# ---np.arange(n)-----np.reshape(x,(-1,8)---------------------------------------------
# import numpy as np
# x = np.arange(16)  #
# print(x)
# print(np.reshape(x,(2,8)))
# print(np.reshape(x,(4,-1)))  # -1 自動分配
# print(np.reshape(x,(-1,8)))
#------------------------------------------------------
# import numpy as np
# x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(f"陣列元素如下 : {x} ")
# print(f"x[2:]       = {x[2:]}")
# print(f"x[:2]       = {x[:3]}")
# print(f"x[0:3]      = {x[0:3]}")
# print(f"x[1:4]      = {x[1:4]}")
# print(f"x[0:9:2]    = {x[0:9:2]}")
# print(f"x[-1]       = {x[-1]}")
# print(f"x[::2]      = {x[::2]}")
# print(f"x[2::3]     = {x[2::3]}")
# print(f"x[:]        = {x[:]}")
# print(f"x[::]       = {x[::]}")
# print(f"x[-3:-7:-1] = {x[-3:-7:-1]}")
#-------------------------------------------------------
# import numpy as np
# x1 = np.array([0, 1, 2, 3, 4, 5])
# x2 = np.array(x1, copy=True)  # copy=True 不影響原來ndarray
# print(x1)
# print(x2)
# print('-'*70)
# x2[0] = 9
# print(x1)
# print(x2)
# print('*'*70)
# x1 = np.array([0, 1, 2, 3, 4, 5])
# x2 = x1.copy() # x1.copy() 不影響原來ndarray
# print(x1)
# print(x1)
# print(x2)
# print('-'*70)
# x2[0] = 9
# print(x1)
# print(x2)
# --------------------------------------------------
# import numpy as np
# x1 = [0, 1, 2, 3, 4]
# x2 = [5, 6, 7, 8, 9]
# x3 = [10, 11, 12, 13, 14]
# x4 = np.array([x1, x2, x3])
# print(x4)
#
# x1 = [0, 1, 2, 3, 4]
# x2 = [5, 6, 7, 8, 9]
# x3 = [10, 11, 12, 13, 14]
# print('-'*30)
# x4 = np.array([x1, x2, x3])
# x5 = np.array([x4, x4])
# print('x5=\n',x5)
#-----------------------------------------
# import numpy as np
# x1 = [0, 1, 2, 3, 4]
# x2 = [5, 6, 7, 8, 9]
# x3 = [10, 11, 12, 13, 14]
# x4 = np.array([x1, x2, x3])
# print(f"x4[2][1] = {x4[2][1]}")
# print(f"x4[1][3] = {x4[1][3]}")
#
# x1 = [0, 1, 2, 3, 4]
# x2 = [5, 6, 7, 8, 9]
# x3 = [10, 11, 12, 13, 14]
# x4 = np.array([x1, x2, x3])
# print(f"x4[2,1] = {x4[2,1]}")
# print(f"x4[1,3] = {x4[1,3]}")
#--------------------------------------------
# import numpy as np
# x1 = [0, 1, 2, 3, 4]
# x2 = [5, 6, 7, 8, 9]
# x3 = [10, 11, 12, 13, 14]
# x4 = np.array([x1, x2, x3])
# x5 = np.array([x4, x4])
# print(f"x5[0][2][1] = {x5[0][2][1]}")
# print(f"x5[0][1][3] = {x5[0][1][3]}")
# print(f"x5[1][0][1] = {x5[1][0][1]}")
# print(f"x5[1][1][4] = {x5[1][1][4]}")
#
# x1 = [0, 1, 2, 3, 4]
# x2 = [5, 6, 7, 8, 9]
# x3 = [10, 11, 12, 13, 14]
# x4 = np.array([x1, x2, x3])
# x5 = np.array([x4, x4])
# print(f"x5[0,2,1] = {x5[0,2,1]}")
# print(f"x5[0,1,3] = {x5[0,1,3]}")
# print(f"x5[1,0,1] = {x5[1,0,1]}")
# print(f"x5[1,1,4] = {x5[1,1,4]}")
#-----------------------------------------
# import numpy as np
# x1 = [0, 1, 2, 3, 4]
# x2 = [5, 6, 7, 8, 9]
# x3 = [10, 11, 12, 13, 14]
# x = np.array([x1, x2, x3])
# print("x[:,:]   = 結果是二維陣列")         # 結果是二維陣列
# print(x[:,:])
# print("-"*70)
# print("x[2,:4]  = 結果是一維陣列")         # 結果是一維陣列
# print(x[2,:4])
# print("-"*70)
# print("x[:2,:1] = 結果是二維陣列")         # 結果是二維陣列
# print(x[:2,:1])
# print("-"*70)
# print("x[:,4:]  =  結果是二維陣列")        # 結果是二維陣列
# print(x[:,4:])
# print("-"*70)
# print("x[:,4]   =  結果是一維陣列")        # 結果是一維陣列
# print(x[:,4])
#
# print("x[:2,4]  = 結果是一維陣列")     # 結果是一維陣列
# print(x[:2,4])
# print("-"*70)
# print("x[:2][4] = 結果是錯誤")         # 結果是錯誤
# print(x[:2][4])
#----------------------------------------------
import numpy as np

x1 = np.arange(4).reshape(2,2)
print(f"陣列 1 \n{x1}")
x2 = np.arange(4,8).reshape(2,2)
print(f"陣列 2 \n{x2}")
x = np.vstack((x1,x2))
print(f"合併結果 \n{x}")

x1 = np.arange(4).reshape(2,2)
print(f"陣列 1 \n{x1}")
x2 = np.arange(4,8).reshape(2,2)
print(f"陣列 2 \n{x2}")
x = np.hstack((x1,x2))
print(f"合併結果 \n{x}")




