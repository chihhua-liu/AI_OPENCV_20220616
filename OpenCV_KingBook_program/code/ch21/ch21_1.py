# ch21_1.py
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft JhengHei"] # 正黑體

seq = [0,1,2,3,4,5,6,7,8,9,10,11]       # 時間值
water = [1,1,1,1,1,1,1,1,1,1,1,1]       # 水
sugar = [2,0,2,0,2,0,2,0,2,0,2,0]       # 糖
grass = [4,0,0,4,0,0,4,0,0,4,0,0]       # 仙草
pearl = [3,0,0,0,3,0,0,0,3,0,0,0]       # 黑珍珠
plt.plot(seq,water,"-o",label="水")     # 繪含標記的water折線圖
plt.plot(seq,sugar,"-x",label="糖")     # 繪含標記的sugar折線圖
plt.plot(seq,grass,"-s",label="仙草")   # 繪含標記的grass折線圖
plt.plot(seq,pearl,"-p",label="黑珍珠") # 繪含標記的pearl折線圖
plt.legend(loc="best")
plt.axis([0, 12, 0, 5])                 # 建立軸大小
plt.xlabel("時間軸")                    # 時間軸
plt.ylabel("份數")                      # 份數
plt.show()


