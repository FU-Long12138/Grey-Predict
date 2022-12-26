# 对序列数据进行事前检验与事后检验
import greypredict.SequenceCheck as GPC
import greypredict.GreyPredictionModel as GPM

Data = [4.93, 2.33, 3.87, 4.35, 6.63, 7.15, 5.37, 6.39, 7.81, 8.35]

# 对所有数据进行准指数检验
GPC.Before_Evaluation(Data, "smooth_check")
# 去除前两个数据进行准指数检验
GPC.Before_Evaluation(Data, "smooth_check", True)
# 对所有数据进行级比检验
GPC.Before_Evaluation(Data, "level_check")

gm11 = GPM.GM11(Data[:-3])
pre_data = gm11.predict(3)

# 使用相对误差进行事后检验
re_error = GPC.After_Evaluation(Data[-3:], pre_data, "Relative Error")
print(f"均方误差为:{re_error}")
# 使用均方差比值进行事后检验
MVR = GPC.After_Evaluation(Data[-3:], pre_data, "Mean Variance Ratio")
print(f"均方误差为:{MVR}")
# 使用小误差概率进行事后检验
PSE = GPC.After_Evaluation(Data[-3:], pre_data, "Probability of Small Error")
print(f"均方误差为:{PSE}")
