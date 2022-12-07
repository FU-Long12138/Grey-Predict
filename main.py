from Grey_Prediction_Model import GM11
import numpy as np
from Sequence_Check import smooth_check
from Sequence_Check import level_check

data = np.sin(np.arange(5, 25, 2))
gm11 = GM11(data)
gm11.model_fit()
pre_data = gm11.predict(3)
print(pre_data)
gm11.evaluation(evaluation_index="Relative Error")
gm11.evaluation(evaluation_index="Mean Variance Ratio")
gm11.evaluation(evaluation_index="Probability of Small Error")
smooth_check(data)
level_check(data)