import numpy as np
import greypredict.GreyPredictionModel as GPM

np.random.seed(0)
data = np.exp(np.arange(0, 1, 0.1)) + 0.1 * np.random.randn(10)
fit_data = data[0:int(len(data)*0.7)]
true_data = data[int(len(data)*0.7):]

gm11 = GPM.GM11(fit_data)
pre_data = gm11.predict(len(true_data))

print(pre_data)
