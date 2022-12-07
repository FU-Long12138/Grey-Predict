from Grey_Prediction_Model import GM11
from Grey_Prediction_Model import NewInformationGM11
from Grey_Prediction_Model import MetabolismGM11
from Grey_Prediction_Model import Verhulst
import numpy as np
from Sequence_Check import Before_Evaluation
from Sequence_Check import After_Evaluation

# data = np.sin(np.arange(5, 25, 2))
X = np.array(
    [21.2, 22.7, 24.36, 26.22, 28.18, 30.16, 32.34, 34.72, 37.3, 40.34, 44.08, 47.92, 51.96, 56.02, 60.14, 64.58, 68.92,
     73.36, 78.98, 86.6]
)
X_train = X[:int(len(X) * 0.7)]
X_test = X[int(len(X) * 0.7):]
Before_Evaluation(X, evaluation_index="level_check")
Before_Evaluation(X, evaluation_index="smooth_check", remove_set=True)
gm11 = GM11(X_train)
pre_data = gm11.predict(6)
print(pre_data)
After_Evaluation(gm11.data, gm11.ori_hat, evaluation_index="Relative Error")
After_Evaluation(gm11.data, gm11.ori_hat, evaluation_index="Mean Variance Ratio")
After_Evaluation(gm11.data, gm11.ori_hat, evaluation_index="Probability of Small Error")
newgm = NewInformationGM11(X_train)
print(newgm.predict(6))
mgm = MetabolismGM11(X_train)
print(mgm.predict(6))
