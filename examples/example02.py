# 使用GM(1,1)模型进行传统短期预测, 新信息短期预测, 新陈代谢短期预测
import numpy as np
import matplotlib.pyplot as plt
import greypredict.GreyPredictionModel as GPM

np.random.seed(0)
x = np.arange(0, 1, 0.1)
data = np.exp(x) + 0.1 * np.random.randn(10)
fit_data = data[0:int(len(data)*0.7)]
true_data = data[int(len(data)*0.7):]

gm11 = GPM.GM11(fit_data)
gm11_data = gm11.predict(len(true_data))

newgm11 = GPM.NewInformationGM11(fit_data)
newgm11_data = newgm11.predict(len(true_data))

metagm11 = GPM.MetabolismGM11(fit_data)
metagm11_data = metagm11.predict(len(true_data))

plt.figure()
plt.plot(x[0:int(len(data)*0.7)], fit_data, label="Ori data")
plt.plot(x[int(len(data)*0.7)-1:], np.append(fit_data[-1], true_data), "r-", label="True data")
plt.plot(x[int(len(data)*0.7)-1:], np.append(fit_data[-1], gm11_data), "r-.", label="GM(1,1)")
plt.plot(x[int(len(data)*0.7)-1:], np.append(fit_data[-1], newgm11_data),
         "g-.", label="New Information GM(1,1)")
plt.plot(x[int(len(data)*0.7)-1:], np.append(fit_data[-1], metagm11_data),
         "b-.", label="Metabolism GM(1,1)")
plt.legend()
plt.grid()
plt.show()
