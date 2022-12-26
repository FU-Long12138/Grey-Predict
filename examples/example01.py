# 使用GM(1,1)模型进行短期预测
import numpy as np
import matplotlib.pyplot as plt
import greypredict.GreyPredictionModel as GPM

np.random.seed(0)
data = np.exp(np.arange(0, 1, 0.1)) + 0.1 * np.random.randn(10)
fit_data = data[0:int(len(data)*0.7)]
true_data = data[int(len(data)*0.7):]

gm11 = GPM.GM11(fit_data)
pre_data = gm11.predict(len(true_data))

plt.figure()
plt.plot(np.arange(0, 1, 0.1)[0:int(len(data)*0.7)], fit_data, "b-", label="Ori data")
plt.plot(np.arange(0, 1, 0.1)[int(len(data)*0.7)-1:], np.append(fit_data[-1], true_data), "r-", label="True data")
plt.plot(np.arange(0, 1, 0.1)[int(len(data)*0.7)-1:], np.append(fit_data[-1], pre_data), "p-.", label="Pre data")
plt.legend()
plt.grid()
plt.show()
