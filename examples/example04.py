# 查看预测模型的各个参数
import numpy as np
import matplotlib.pyplot as plt
import greypredict.GreyPredictionModel as GPM

data = [4.93, 5.33, 5.87, 6.35, 6.63, 7.15, 7.37, 7.39, 7.81, 8.35, 9.39]
gm11 = GPM.GM11(data)
gm11.model_fit()
a = gm11.a
b = gm11.b
print(f"发展系数a为:{a}, 灰作用变量b为{b}")
AGO = gm11.AGO
Z = gm11.Z
print(f"累加生成序列为{AGO}")
print(f"紧邻生成序列为{Z}")

gm11.predict(0)
# 灰色预测模型对数据的拟合值
hat_data = gm11.ori_hat

plt.figure()
plt.plot(np.arange(1, len(data)+1), data, "b-", label="Ori data")
plt.plot(np.arange(1, len(data)+1), hat_data, "r-.", label="True data")
plt.legend()
plt.grid()
plt.show()
